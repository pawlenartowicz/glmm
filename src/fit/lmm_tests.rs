//! LMM estimator tests (`Family::Gaussian`, `re: Some`).

use super::*;
use crate::standardize::ColumnScaling;
// The loop-tier entries reached through the module rather than the re-export,
// which is `loop_advanced`-gated: the pivot the region-2 tests below assert on
// is recorded on every route, so these must run under default features too.
use super::core::{build_workspace, fit_on};
use crate::lmm::{fit_lmm_unhinted, LmmWorkspace};
use crate::{
    Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing, StartValues,
};
use faer::Mat;

use super::common_tests::{assert_pinned, dense_str, lcg, PIN_REL_ITER};

use super::lmm::{accumulate_lmm_rows, lmm_run_on, lmm_view_to_fit};
use crate::test_support::{cluster_of_row, extra_level_of_row, intercept_only_spec};

/// Aliased fixed column on a MIXED design: `y ~ 1 + x1 + x2 + x3 + (1|g)` on
/// sim_collinear_lmm (x3 ≈ x1 + x2). glmm keeps full width with `NaN` in the
/// dropped slot and flags it in `aliased`; the rest of the fit is the reduced
/// model, varcomp included — the salvage must not perturb θ.
///
/// Values recorded from glmm. They are validated by `sim_collinear_lmm`, whose
/// cross-engine cell checks the same fit against lme4 and asserts the two
/// engines drop the SAME column — lmer's rankMatrix check omits the name from
/// `fixef` entirely, so that comparison has to align by name and belongs there.
#[test]
fn fit_lmm_rank_deficient_drops_the_aliased_column() {
    // Surviving coefficients of the reduced fit; index 3 is the dropped x3.
    const REF_BETA: [f64; 3] = [0.8576729942296913, 0.6993983638391031, -0.4068182431411529];
    const REF_SE: [f64; 3] = [
        0.24654045945855108,
        0.041312856909260794,
        0.042805742152106876,
    ];
    // tau2 and dispersion are the variance scales, not stddev/sigma.
    const REF_G_TAU2: f64 = 0.7113844334703112;
    const REF_SIGMA2: f64 = 0.26968316460592023;

    // sim_collinear_lmm.csv: y,x1,x2,x3,g
    let csv = include_str!("../../validation/data/simulated/sim_collinear_lmm.csv");
    let mut y = Vec::<f64>::new();
    let mut cols: Vec<[f64; 3]> = Vec::new();
    let mut g_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        cols.push([
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        ]);
        g_raw.push(f[4].to_string());
    }
    let n = y.len();
    let p = 4; // intercept + x1 + x2 + x3
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = cols[i][0];
        x[i * p + 2] = cols[i][1];
        x[i * p + 3] = cols[i][2];
    }
    let (g, _n_g) = dense_str(&g_raw);
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: g,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "reduced LMM must converge");
    assert_eq!(
        f.aliased(),
        vec![false, false, false, true],
        "x3 is the dependent column and the only one dropped"
    );
    assert!(f.beta[3].is_nan(), "aliased β = NaN");
    assert!(f.se[3].is_nan(), "aliased se = NaN");
    assert_pinned(&f.beta[..3], &REF_BETA, PIN_REL_ITER, "beta");
    assert_pinned(&f.se[..3], &REF_SE, PIN_REL_ITER, "se");
    // Varcomp of the reduced fit passes through the salvage unchanged.
    assert_pinned(&f.tau2, &[REF_G_TAU2], PIN_REL_ITER, "tau2");
    assert_pinned(&[f.dispersion], &[REF_SIGMA2], PIN_REL_ITER, "sigma2");
}

// ---------------------------------------------------------------------------
// Ill-conditioned but computable — the designs that must be FITTED, not refused
//
// Both designs below clear the alias gate (`detect_aliased`, X'X,
// `ALIAS_EPS = 1e-14`): nothing in them is redundant in f64, so there is no
// column whose removal is the right answer. They are merely badly conditioned,
// which means the fit is computable and unique and the honest expression of the
// imprecision is a large standard error. The dense-LMM route refuses no
// design on conditioning grounds: `min|L_ii| / max|L_ii|` on X'V⁻¹X measures
// column SCALE rather than collinearity, so it is not used as a rejection
// statistic. Instead the route records the scale-invariant per-column pivot
// ratio for the diagnostics channel to flag below `lmm::PIVOT_MIN = 1e-12`.
// Neither design reaches even that.
//
// Designs generated from a 16-bit LCG (`s <- (75s + 74) mod 65537`, value
// `s/65537 - 0.5`); every intermediate is below 2^53, so the stream is exact in
// f64 and the R reference builder reproduces it bit for bit.
// ---------------------------------------------------------------------------

/// `s_{k+1} = (75·s_k + 74) mod 65537`, value `s/65537 − 0.5`. Local to the Gap A
/// designs (the shared `lcg` helper is a different, 64-bit generator).
fn gap_a_stream(k: usize) -> Vec<f64> {
    let mut s = 1u64;
    (0..k)
        .map(|_| {
            s = (75 * s + 74) % 65537;
            s as f64 / 65537.0 - 0.5
        })
        .collect()
}

fn intercept_only_lmm() -> ModelSpec {
    // Placeholder sizing — `spec_sized_from_ids` derives the real count.
    intercept_only_spec(Sizing::FixedClusters { n_clusters: 1 })
}

/// PURE DYNAMIC RANGE — no collinearity anywhere, and the fit must come out
/// whole. `y ~ 1 + u + w + (1|g)`, J=25 × m=40. `u` is CLUSTER-LEVEL at scale
/// 3e-7, `w` is WITHIN-CLUSTER mean-zero at scale 20. V⁻¹ divides the
/// cluster-level block by `sqrt(1 + m·λ²)` (λ̂ ≈ 14.7 ⇒ 93×) and leaves the
/// within-cluster block alone, so the min/max L-diagonal ratio of X'V⁻¹X lands
/// at 1.6e-10 while X'X's own ratio (1.5e-8) is four orders clear of even the
/// OLS guard.
///
/// This is the control that condemned the old statistic. Every per-column pivot
/// ratio here is O(1) — the columns are mutually
/// distinguishable to full precision — yet the min/max L-diagonal ratio sits at
/// 1.6e-10 purely because one column is 8 orders smaller than another. The old
/// guard therefore threw the whole fit away over a choice of units: rescaling
/// `u` alone moved its statistic by six decades while β̂ did not move by one part
/// in 1e10 (measured 2026-07-31 across `c` = 1e-4 … 1e-10).
///
/// So the fit must converge with all three columns, and `u`'s coefficient must
/// carry an enormous standard error — that SE is the correct report on a column
/// whose entries are 3e-7, not a defect. The signal is well-conditioned even
/// though the COLUMN is tiny (β_u enters `y` as 2 on `u/c`), so the estimate
/// must also stay within a standard error of the truth `2/c`.
///
/// The assertions below are against the data-generating truth, which is what a
/// default-tier test can check without a reference. The cross-engine check on
/// the same fit is the `sim_dynrange_lmm` golden: this design is emitted
/// bit-identically as `validation/data/simulated/sim_dynrange_lmm.csv` by
/// `validation/tools/prep/gen_illcond_data.R`, and `tests/validation_oracle.rs` bands
/// its β, SE, σ̂, log-likelihood and variance components against lme4 on the FULL
/// three-column design at `validation/grid/tol.R`'s cross-engine tolerances. That
/// golden is possible because the fit returns all three columns rather than
/// NaN-filling one: a NaN-filled column would leave nothing for a reference to
/// agree with. The two engines land within 6e-11 on β
/// and 1.1e-7 on the SEs — this design is ill-conditioned in the old statistic's
/// eyes only, and both engines say so.
#[test]
fn lmm_pure_dynamic_range_design_fits_in_full() {
    let (jn, m, c, s_scale, tau, sigma) = (25usize, 40usize, 3e-7f64, 20.0f64, 16.0f64, 1.0f64);
    let (n, p) = (jn * m, 3usize);
    let g = gap_a_stream(jn);
    let h = gap_a_stream(n);
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for (j, &g_j) in g.iter().enumerate().take(jn) {
        let u_j = c * ((j + 1) as f64 / jn as f64);
        let b_j = tau * g_j;
        for i in 0..m {
            let r = j * m + i;
            let w_i = s_scale * (i as f64 / m as f64 - (m as f64 - 1.0) / (2.0 * m as f64));
            x[r * p] = 1.0;
            x[r * p + 1] = u_j;
            x[r * p + 2] = w_i;
            // u enters the response at unit scale (β_u = 2 on u/c), so the
            // signal is well-conditioned even though the COLUMN is tiny.
            y[r] = 1.0 + 2.0 * (u_j / c) + 0.5 * w_i + b_j + sigma * h[r];
            ids[r] = j as u32;
        }
    }
    let ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1, 2],
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, p, &intercept_only_lmm(), &ids, &opts);

    assert!(
        f.converged(),
        "nothing is collinear here — the design is computable and must fit"
    );
    assert_eq!(
        f.aliased(),
        vec![false; p],
        "nothing was dropped, so nothing may be flagged aliased"
    );
    assert!(
        f.beta.iter().all(|b| b.is_finite()) && f.se.iter().all(|s| s.is_finite()),
        "the full fit must be finite throughout, got β = {:?}, se = {:?}",
        f.beta,
        f.se
    );
    // The scale-driven imprecision lands where it belongs: on `u`'s SE, which is
    // ~1e7 because the column's entries are ~3e-7. The two ordinary columns keep
    // ordinary SEs, so the report is not "everything is uncertain".
    assert!(
        f.se[1] > 1e6,
        "u's SE must carry the imprecision, got {}",
        f.se[1]
    );
    assert!(
        f.se[0] < 10.0 && f.se[2] < 1.0,
        "the well-scaled columns keep ordinary SEs, got {:?}",
        f.se
    );
    // β_u is estimable to within its own SE of the DGP truth 2/c — the fit is
    // imprecise, not wrong.
    assert!(
        (f.beta[1] - 2.0 / c).abs() < f.se[1],
        "β_u = {} must sit within one SE ({}) of the truth {}",
        f.beta[1],
        f.se[1],
        2.0 / c
    );
    // `w` is the well-conditioned column; its coefficient is pinned tightly.
    assert!(
        (f.beta[2] - 0.5).abs() < 0.01,
        "β_w = {} must recover the truth 0.5",
        f.beta[2]
    );
    assert!(f.df > 0, "a converged fit reports its parameter count");

    // The flag itself. This design is the control that condemned the old
    // statistic, so the verdict under the NEW one must be "not flagged": no
    // note on the stable surface.
    assert!(
        f.diagnostics.notes.is_empty(),
        "no column is entangled here, so no note may be raised: {:?}",
        f.diagnostics.notes
    );
}

/// The entangled-pair design, shared by the two tests below.
/// Returns `(x, y, ids, n, p)`.
///
/// `y ~ 1 + t + v + z [+ s] + (1|g)`, J=25 × m=40, every predictor within-cluster
/// mean-zero:
///   * `t` at scale `20·rho` (rho = 1e-5) — a mean-centred LCG pattern
///   * `v = t·(1 + d·(−1)^i)` (d = 3e-6) — near-collinear with `t`, but four
///     orders clear of `ALIAS_EPS`, so it is entangled with `t`, not redundant
///     with it, and no gate drops it
///   * `z` at scale 20 — the ramp; sets the max L-diagonal
///   * `s = 1 + z` (only when `with_exact_alias`) — EXACTLY dependent on columns
///     0 and 3, so `detect_aliased` catches it at `ALIAS_EPS` and the alias gate
///     fires before the solver ever runs
///
/// With `with_exact_alias = true` the leading four columns are BIT-IDENTICAL to
/// the `false` design, so the second test's post-drop fit is the same fit the
/// first test performs — the two tests' numbers must agree exactly.
fn build_gap_a_salvage_design(
    with_exact_alias: bool,
) -> (Vec<f64>, Vec<f64>, Vec<u32>, usize, usize) {
    let (jn, m, s_scale, tau, sigma) = (25usize, 40usize, 20.0f64, 16.0f64, 1.0f64);
    let (d, rho) = (3e-6f64, 1e-5f64);
    let n = jn * m;
    let p = if with_exact_alias { 5 } else { 4 };
    let s_small = s_scale * rho;
    let g = gap_a_stream(jn);
    // One stream, split: h[..n] shapes `t`, h[n..] is the residual noise. Reusing
    // the same slice for both would make the noise collinear with `t` and drive
    // σ̂² to zero.
    let h = gap_a_stream(2 * n);
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for (j, &g_j) in g.iter().enumerate().take(jn) {
        let b_j = tau * g_j;
        let t_bar = h[j * m..j * m + m].iter().sum::<f64>() / m as f64;
        for i in 0..m {
            let r = j * m + i;
            // z: the ramp. t: a DIFFERENT within-cluster mean-zero pattern, so t
            // and z are not collinear with each other — only t and v are.
            let z_i = s_scale * (i as f64 / m as f64 - (m as f64 - 1.0) / (2.0 * m as f64));
            let t_i = s_small * (h[r] - t_bar);
            let v_i = t_i * (1.0 + d * if i % 2 == 0 { 1.0 } else { -1.0 });
            x[r * p] = 1.0;
            x[r * p + 1] = t_i;
            x[r * p + 2] = v_i;
            x[r * p + 3] = z_i;
            if with_exact_alias {
                x[r * p + 4] = 1.0 + z_i;
            }
            y[r] = 1.0 + (1.0 / s_small) * t_i + 0.5 * z_i + b_j + sigma * h[n + r];
            ids[r] = j as u32;
        }
    }
    (x, y, ids, n, p)
}

/// ENTANGLED PAIR — distinguishable in f64, so the full model is what comes
/// back. `y ~ 1 + t + v + z + (1|g)`, J=25 × m=40, all three predictors
/// WITHIN-CLUSTER mean-zero:
///   * `t` at scale `20·rho` (rho = 1e-5),
///   * `v = t·(1 + d·(−1)^i)` (d = 3e-6) — near-collinear with `t`,
///   * `z` at scale 20 — the large column that sets the max L-diagonal.
///
/// Three measured margins, all ≥ 100× (the flakiness floor from the sensitivity
/// analysis: the alias pivot is a cancelled quantity, and this crate already
/// carries pin failures from that class of FP drift):
///   * alias gate must NOT trip — `v`'s X'X pivot ratio 8.98e-12 vs 1e-14: 898×
///   * the design is not even flagged — `v`'s pivot ratio in X'V⁻¹X is 8.76e-12,
///     8.76× above `lmm::PIVOT_MIN`
///   * the old min/max L-diag statistic sat at 2.96e-11, 338× INSIDE the old
///     `EPS_RANK`, and threw the fit away — the whole gap between the two
///     verdicts on one design
///
/// `t` and `v` are not separately identified to any useful precision, and the
/// fit says so: each gets a coefficient near ±3.8e7 with a standard error of the
/// same size. That is the deliverable. It is also where lme4 lands — it
/// fits all four columns of this design with the same ±3.8e7 blow-up — so the
/// crate's four-column fit agrees with the reference on the column set.
///
/// The pair sits in the within-cluster block deliberately. V⁻¹ downdates the
/// cluster-level block with per-cluster outer products, and on a near-collinear
/// CLUSTER-LEVEL pair that downdate cancels away the pivot entirely: the
/// deviance re-eval at θ̂ returns infinity and the rank guard never runs, so such
/// a design is not a witness for this branch at all. The within-cluster block is
/// untouched by the downdate.
///
/// Reference values are lme4's on the FULL four-column design — the same design
/// this test fits, column for column. That is what makes the entangled pair
/// itself assertable: the four-column fit matches lme4's four-column
/// reference directly, with no need to band only the unentangled columns and
/// the identified sum against a reduced design. Bands are `validation/grid/tol.R`'s
/// cross-engine ones, unchanged.
///
/// The provenance, since these constants are frozen in-crate rather than under
/// `validation/goldens/`: the design is emitted as
/// `validation/data/simulated/sim_entangled_pair_lmm.csv` by
/// `validation/tools/prep/gen_illcond_data.R`, whose generator arithmetic is
/// bit-identical to [`build_gap_a_salvage_design`] above (verified over all 4000
/// doubles) and whose CSV round-trips exactly at 17 significant digits. The
/// reference is `lmer(y ~ 1 + t + v + z + (1|g), data, REML = TRUE)` under lme4
/// 1.1.38 / R 4.5.3.
///
/// It is NOT registered in `validation/manifest.json` as a cross-engine golden,
/// and the reason is worth stating rather than leaving to be rediscovered. Every
/// quantity below agrees inside its band, but lme4's REML criterion does not
/// match the true one: lme4 reports −239.09437 against the exact −239.0944407
/// (the design's closed form for one balanced intercept RE evaluated in
/// 60-digit arithmetic), off by +7.0e-5 where `grid/tol.R`'s `loglik_abs_lmm`
/// is an absolute 2e-6. The θ̂ are not what separates them — the exact
/// criterion is flat to 1e-10 across both engines' θ̂ — so the gap is lme4's
/// round-off in evaluating the criterion on this nearly collinear X. glmm
/// evaluates `log|X'V⁻¹X|` on X·U⁻¹ (`lmm::DesignQr`) and lands on the exact
/// value, which
/// `lmm_entangled_pair_reml_is_exact_and_invariant` pins. The reference is the
/// side a cross-engine band would have to absorb, so registering the rung needs
/// the band question settled first, which is a calibration decision, not a test
/// edit.
#[test]
fn lmm_entangled_pair_fits_in_full_with_honest_ses() {
    // lme4 1.1.38 on the FULL design [1, t, v, z], REML. Written in the shortest
    // decimal form that round-trips to the same f64 as lme4's 17-digit output —
    // the same doubles, not truncated ones; padding them back out is a clippy
    // `excessive_precision` error and changes nothing. Measured agreement with
    // glmm, worst per row: β 1.4e-4 (both entangled columns), SE 7.0e-5 (same
    // two), stddev 4.7e-8, σ̂ 1.2e-9, β_t + β_v 1.2e-7. The pair is by
    // construction the least-determined direction in the design, so it is where
    // lme4's round-off shows most; glmm's β_t is within 1e-11 of the 60-digit
    // value (`lmm_entangled_pair_reml_is_exact_and_invariant`).
    const LME4_BETA: [f64; 4] = [
        -0.7054541628205219,
        -38288906.83665362,
        38293871.58172187,
        0.5016368546595636,
    ];
    const LME4_SE: [f64; 4] = [
        0.8424257561779566,
        52060999.05491157,
        52060993.35174688,
        0.0015371530739338938,
    ];
    const LME4_SD_G: f64 = 4.211895307851695;
    const LME4_SIGMA: f64 = 0.2803066654730708;
    // validation/grid/tol.R: beta_rel, se_rel, stddev_rel.
    const BETA_REL: f64 = 1e-3;
    const SE_REL: f64 = 1e-3;
    const STDDEV_REL: f64 = 1e-3;

    let (x, y, ids, n, p) = build_gap_a_salvage_design(false);
    let opts = FitOptions {
        target_indices: (0..p as u32).collect(),
        ..FitOptions::default()
    };
    let ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let f = fit_cold(&x, &y, n, p, &intercept_only_lmm(), &ids, &opts);

    assert!(
        f.converged(),
        "the design is ill-conditioned, not rank-deficient — it must fit"
    );
    assert_eq!(
        f.aliased(),
        vec![false; 4],
        "nothing is redundant at ALIAS_EPS, so no column may be dropped"
    );
    assert!(
        f.beta.iter().all(|b| b.is_finite()) && f.se.iter().all(|s| s.is_finite()),
        "the full fit must be finite throughout, got β = {:?}, se = {:?}",
        f.beta,
        f.se
    );
    // The entangled pair reports its own imprecision: |β| ≈ 3.8e7 with an SE of
    // the same order, i.e. neither coefficient is distinguishable from zero.
    for j in [1usize, 2] {
        assert!(
            f.se[j] > 0.5 * f.beta[j].abs(),
            "β[{j}] = {} must carry an SE of its own size, got {}",
            f.beta[j],
            f.se[j]
        );
    }
    // EVERY column against lme4 on the same four-column design, the entangled
    // pair included. This is the assertion the reduced-design reference could
    // not make.
    assert_pinned(&f.beta, &LME4_BETA, BETA_REL, "beta vs lme4 full design");
    assert_pinned(&f.se, &LME4_SE, SE_REL, "se vs lme4 full design");
    // The identified combination gets its own line because it is a far better
    // determined quantity than either coefficient: v = t·(1 + d·(−1)^i), so
    // β_t + β_v is what the data actually pins, and the two engines agree on it
    // to 4.8e-7 while agreeing on its two summands only to 5.7e-4. Asserting it
    // separately keeps that three-order gap under test — a regression that moved
    // both coefficients together would slip past the per-column bands.
    assert_pinned(
        &[f.beta[1] + f.beta[2]],
        &[LME4_BETA[1] + LME4_BETA[2]],
        BETA_REL,
        "β_t + β_v vs lme4 full design",
    );
    assert_eq!(f.tau2.len(), 1, "one variance component, got {:?}", f.tau2);
    // Compare on the STDDEV scale, which is what grid/tol.R's stddev_rel bands.
    assert_pinned(
        &[f.tau2[0].sqrt(), f.dispersion.sqrt()],
        &[LME4_SD_G, LME4_SIGMA],
        STDDEV_REL,
        "stddevs vs lme4 full design",
    );

    // The flag itself. The doc comment above turns on this design sitting just
    // ABOVE the detection floor — 8.76e-12 against `lmm::PIVOT_MIN` — and
    // nothing else in this test would notice if that stopped being true.
    assert!(
        f.diagnostics.notes.is_empty(),
        "the pair is distinguishable in f64, so no note may be raised: {:?}",
        f.diagnostics.notes
    );
    let (sized, ids, perm) = spec_sized_from_ids_pub(&intercept_only_lmm(), &ids);
    let mut ws = build_workspace(&sized, perm, n, p, &opts);
    let d = fit_on(&mut ws, &x, &y, &ids, None, &opts).diagnostics();
    assert!(!d.ill_conditioned, "pivot {} must clear the floor", d.pivot);
    // A 2× band, not a pin. This pivot is a cancelled quantity and the doc's
    // own margins are stated at 100×, so banding it tighter would buy a flaky
    // test; banding it at all keeps the quoted decade under test.
    assert!(
        (4.4e-12..1.8e-11).contains(&d.pivot),
        "min pivot ratio must stay at the quoted 8.76e-12, got {}",
        d.pivot
    );

    // 1-ULP stability: the guard's promise is that a fit it ACCEPTS still has
    // significant digits left. Re-round every entry of `y` by one ULP and refit;
    // the identified quantities must not move. Perturbing a SINGLE double is
    // useless at n = 1000 — the Gram accumulation absorbs it exactly and reports
    // a spurious zero — so every entry moves.
    //
    // Two alternating sign patterns, not the calibration's worst-of-16
    // pseudorandom ones: alternating signs cancel heavily in the accumulation,
    // so this is a weaker probe than the 2026-07-31 measurement and must not be
    // read as reproducing its `betaRel`. It is a tripwire against a guard placed
    // low enough to accept arithmetic noise, where the movement would be O(1).
    for flip in [false, true] {
        let y_eps: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                // ±1 ULP by direct bit step. Sign-magnitude layout: for v > 0 a
                // larger bit pattern is a larger value, for v < 0 the reverse.
                let up = (i % 2 == 0) != flip;
                let bits = v.to_bits();
                if v == 0.0 {
                    v
                } else if v.is_sign_positive() == up {
                    f64::from_bits(bits + 1)
                } else {
                    f64::from_bits(bits - 1)
                }
            })
            .collect();
        let g = fit_cold(&x, &y_eps, n, p, &intercept_only_lmm(), &ids, &opts);
        assert!(
            g.converged(),
            "flip={flip}: the perturbed fit must also converge"
        );
        // Measured worst across both patterns: 3.7e-12 on the unentangled
        // columns, 7.5e-12 on β_t + β_v. The band is five orders above that, so
        // it does not fail on cross-platform FP drift, and six orders below a
        // fit that has lost its digits.
        const ULP_REL: f64 = 1e-6;
        for j in [0usize, 3] {
            let rel = (g.beta[j] - f.beta[j]).abs() / f.beta[j].abs();
            assert!(
                rel < ULP_REL,
                "flip={flip}: β[{j}] moved {rel} under a 1-ULP re-rounding of y"
            );
        }
        let sum = f.beta[1] + f.beta[2];
        let rel = ((g.beta[1] + g.beta[2]) - sum).abs() / sum.abs();
        assert!(
            rel < ULP_REL,
            "flip={flip}: β_t + β_v moved {rel} under a 1-ULP re-rounding of y"
        );
    }
}

/// The entangled pair's REML criterion and β̂ at the exact values, and unmoved
/// by an exact reparametrization of the pair.
///
/// Replacing `v` by `w = v − t` is `X·A` with `A` unit upper triangular, so the
/// REML criterion is unchanged (`log|det A| = 0`) and `β_t = β′_t − β′_v`,
/// `β_v = β′_v`. The subtraction is exact in f64 (Sterbenz: `v/t` is within
/// `1 ± 3e-6`), so both fits see the same model on the same doubles. `[1, t, w,
/// z]` is well conditioned and `[1, t, v, z]` is not, which makes the pair a
/// probe of how the kernel evaluates `log|X'V⁻¹X|` and β̂: a Gram of the raw
/// `X` squares its condition number and moves the criterion by 6.5e-4 between
/// the two, while the orthogonalized route (`lmm::DesignQr`) agrees to 3e-10.
///
/// References: the exact criterion is `validation/grid/dev_ref.json`'s
/// 478.188881405005 (60-digit arithmetic on the same doubles, flat to 1e-10
/// across every engine's θ̂), on the grid's scale, which is `Fit::deviance` plus
/// the REML constant `(n − p)·(1 + ln 2π)`. The exact β_t, −38283544.440675,
/// is the same 60-digit evaluation
/// (`validation/tools/prep/entangled_reml_reference.py`'s method) at this fit's
/// θ̂ = 15.0260270.
#[test]
fn lmm_entangled_pair_reml_is_exact_and_invariant() {
    const EXACT_REML_CRIT: f64 = 478.188881405005;
    const EXACT_BETA_T: f64 = -38283544.440675;

    let (x, y, ids, n, p) = build_gap_a_salvage_design(false);
    let mut x_w = x.clone();
    for r in 0..n {
        x_w[r * p + 2] = x[r * p + 2] - x[r * p + 1];
    }
    let opts = FitOptions {
        target_indices: (0..p as u32).collect(),
        ..FitOptions::default()
    };
    let ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let f = fit_cold(&x, &y, n, p, &intercept_only_lmm(), &ids, &opts);
    let g = fit_cold(&x_w, &y, n, p, &intercept_only_lmm(), &ids, &opts);
    assert!(
        f.converged() && g.converged(),
        "both parametrizations must fit"
    );

    // Measured: 2.4e-10 between the two, 1.9e-10 against the 60-digit criterion
    // at this θ̂. The Gram route missed both by 6.5e-4.
    assert!(
        (f.deviance - g.deviance).abs() < 1e-8,
        "REML deviance moved under an exact reparametrization: {} vs {}",
        f.deviance,
        g.deviance
    );
    let reml_const = (n - p) as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (f.deviance + reml_const - EXACT_REML_CRIT).abs() < 1e-8,
        "REML criterion {} vs exact {EXACT_REML_CRIT}",
        f.deviance + reml_const
    );
    // Measured: 8.7e-12 against the 60-digit β_t; 7.5e-10 between the two
    // parametrizations, whose θ̂ differ by 5e-7 relative (β̂ follows θ̂). The
    // Gram route was 4.2e-4 off. 1e-6 leaves room for θ̂ to land elsewhere in
    // BOBYQA's stopping radius on another CPU.
    assert_pinned(&[f.beta[1]], &[EXACT_BETA_T], 1e-6, "β_t vs 60-digit value");
    assert_pinned(
        &[f.beta[1], f.beta[2]],
        &[g.beta[1] - g.beta[2], g.beta[2]],
        1e-6,
        "β_t, β_v under v → v − t",
    );
}

/// REDUNDANCY AND ENTANGLEMENT IN ONE DESIGN — the two must be told apart.
///
/// The design above plus `s = 1 + z`, exactly dependent on columns 0 and 3. `s`
/// is genuinely redundant: there is no separate coefficient for it, so
/// `detect_aliased` fires at `ALIAS_EPS` before the solver runs, the column is
/// dropped, and `fit_warm` re-enters on a reduced design whose four columns are
/// bit-identical to the single-level design. Those four are merely entangled,
/// and the reduced fit keeps every one of them.
///
/// So this pins the discrimination the whole guard rework is about: exactly one
/// column is dropped from a design that contains both an exact dependency and a
/// near one, and `aliased` is `[f,f,f,f,t]` rather than flagging `v` too.
///
/// It also pins that the drop does not perturb the numbers: the inner fit IS the
/// single-level test's fit, so β/se/τ/σ must land on the same lme4 reference
/// values, and `tau2` must keep its width (no RE block is ever dropped).
#[test]
fn exact_alias_is_dropped_and_the_entangled_pair_is_kept() {
    // Same lme4 1.1.38 FULL-design reference as
    // `lmm_entangled_pair_fits_in_full_with_honest_ses` — the fit reached after
    // `s` is dropped is that same four-column fit, so the same four references
    // apply and the entangled pair is assertable here too. Provenance is
    // recorded at that test.
    const LME4_BETA: [f64; 4] = [
        -0.7054541628205219,
        -38288906.83665362,
        38293871.58172187,
        0.5016368546595636,
    ];
    const LME4_SE: [f64; 4] = [
        0.8424257561779566,
        52060999.05491157,
        52060993.35174688,
        0.0015371530739338938,
    ];
    const LME4_SD_G: f64 = 4.211895307851695;
    const LME4_SIGMA: f64 = 0.2803066654730708;
    const BETA_REL: f64 = 1e-3;
    const SE_REL: f64 = 1e-3;
    const STDDEV_REL: f64 = 1e-3;

    let (x, y, ids, n, p) = build_gap_a_salvage_design(true);
    assert_eq!(p, 5);
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &intercept_only_lmm(),
        &GroupIds {
            primary: ids,
            extra: vec![],
        },
        &FitOptions {
            target_indices: (0..p as u32).collect(),
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "the reduced fit must converge");
    assert_eq!(
        f.aliased(),
        vec![false, false, false, false, true],
        "only the EXACT dependency (s, index 4) is dropped; the near-collinear \
         pair (t, v) is entangled, not redundant, and both columns stay"
    );
    // The contract `Fit::aliased` exists to carry, asserted directly rather than
    // only through the mask: NaN in β/se iff flagged aliased, for every column.
    for j in 0..p {
        assert_eq!(
            f.beta[j].is_nan(),
            f.aliased()[j],
            "β[{j}] = {} but aliased[{j}] = {}",
            f.beta[j],
            f.aliased()[j]
        );
        assert_eq!(
            f.se[j].is_nan(),
            f.aliased()[j],
            "se[{j}] = {} but aliased[{j}] = {}",
            f.se[j],
            f.aliased()[j]
        );
    }
    // Dropping `s` must not move the rest: the inner fit is the single-level
    // test's fit, so the same lme4 reference applies, read the same way — every
    // surviving column directly, plus the identified sum.
    assert_pinned(
        &f.beta[..4],
        &LME4_BETA,
        BETA_REL,
        "reduced beta vs lme4 full design",
    );
    assert_pinned(
        &f.se[..4],
        &LME4_SE,
        SE_REL,
        "reduced se vs lme4 full design",
    );
    assert_pinned(
        &[f.beta[1] + f.beta[2]],
        &[LME4_BETA[1] + LME4_BETA[2]],
        BETA_REL,
        "β_t + β_v vs lme4 full design",
    );
    assert_eq!(f.tau2.len(), 1, "one variance component, got {:?}", f.tau2);
    assert_pinned(
        &[f.tau2[0].sqrt(), f.dispersion.sqrt()],
        &[LME4_SD_G, LME4_SIGMA],
        STDDEV_REL,
        "nested stddevs vs lme4 full design",
    );
}

/// Warm-start A/B on the realistic sleepstudy random-slope LMM
/// (`Reaction ~ Days + (1 + Days | Subject)`, q=2, n_theta=3): a warm fit
/// from the frozen lme4 θ̂ ("from the truth") and one from a well-off
/// perturbed θ must land on the cold optimum — β, SE, and the varcorr
/// stddevs — and warm must never degrade convergence status. Extends
/// `fit_warm_start_reaches_cold_beta` (β-only, hand-built n_theta=1) to a
/// realistic q≥2 rung; MCPower's hot loop rides this contract.
#[test]
fn fit_warm_sleepstudy_slope_matches_cold_optimum() {
    // Parsing mirrors `fit_sleepstudy_slope_varcorr_matches_lme4`.
    let (x, y, n, p, model, ids) = sleepstudy_slope_design();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(cold.converged(), "cold sleepstudy fit must converge");

    // lme4 θ̂ = vech Cholesky of D̂/σ̂², from the frozen golden's
    // stddev/corr/sigma (`validation/goldens/sleepstudy_lmm.json`) — `Fit`
    // does not expose θ̂, and Gaussian tau2/varcorr are both σ²-scaled so
    // θ cannot be recovered from the cold fit alone:
    // θ00 = sd0/σ, θ10 = corr·sd1/σ, θ11 = (sd1/σ)·√(1−corr²).
    const REF_SD0: f64 = 24.7406579949841;
    const REF_SD1: f64 = 5.92213765889808;
    const REF_CORR: f64 = 0.0655512382381282;
    const REF_SIGMA: f64 = 25.5917957216753;
    let truth = vec![
        REF_SD0 / REF_SIGMA,
        REF_CORR * REF_SD1 / REF_SIGMA,
        REF_SD1 / REF_SIGMA * (1.0 - REF_CORR * REF_CORR).sqrt(),
    ];
    let starts = [
        (
            "truth",
            StartValues {
                beta: cold.beta.clone(),
                theta: truth,
                dispersion: None,
            },
        ),
        // Well off θ̂ ≈ [0.97, 0.015, 0.23] in every coordinate; the LMM
        // path threads θ only (β is solved exactly given θ).
        (
            "perturbed",
            StartValues {
                beta: vec![0.0; p],
                theta: vec![3.0, 0.5, 1.5],
                dispersion: None,
            },
        ),
    ];
    for (label, start) in &starts {
        let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(start), &opts);
        assert!(
            warm.converged(),
            "{label}: warm must not degrade convergence"
        );
        for j in 0..p {
            let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
            assert!(
                rel < 1e-3,
                "{label}: β[{j}] warm {} vs cold {} (rel {rel})",
                warm.beta[j],
                cold.beta[j]
            );
            let rel = (warm.se[j] - cold.se[j]).abs() / cold.se[j];
            assert!(
                rel < 1e-3,
                "{label}: se[{j}] warm {} vs cold {} (rel {rel})",
                warm.se[j],
                cold.se[j]
            );
        }
        // q=2 vech diag (offsets 0, 2) → the two RE stddevs. The off-diag
        // covariance is near zero here (corr≈0.066); the stddevs pin the block.
        for off in [0usize, 2] {
            let (w, c) = (warm.varcorr[0][off].sqrt(), cold.varcorr[0][off].sqrt());
            let rel = (w - c).abs() / c;
            assert!(
                rel < 1e-3,
                "{label}: RE stddev (vech {off}) warm {w} vs cold {c} (rel {rel})"
            );
        }
    }
}

/// Gap #3: sleepstudy `Reaction ~ 1 + Days + (1 + Days | Subject)` — a q=2
/// random-slope LMM through `fit_cold`, gated against the frozen lme4 VarCorr
/// (`validation/goldens/sleepstudy_lmm.json`, REML). Checks the full 2×2 RE
/// covariance (variances AND the off-diagonal covariance) via `Fit::varcorr`,
/// which `tau2` cannot represent at q≥2. The oracle is sacred.
///
/// The same fit must also surface the optimizer eval count, the minimized
/// criterion, and boundary/singular status. Oracle: lme4's frozen REML fit —
/// REMLcrit = glmm deviance + df·(1 + ln 2π), df = n − p (glmm's reml_deviance
/// omits the df·(1+ln 2π) constant lme4's REMLcrit carries; loglik =
/// −REMLcrit/2 is what `validation/goldens/sleepstudy_lmm.json` stores).
#[test]
fn fit_sleepstudy_slope_varcorr_matches_lme4() {
    const REF_B0: f64 = 251.405104848485;
    const REF_B1: f64 = 10.467285959596;
    const REF_SE0: f64 = 6.82459669495491;
    const REF_SE1: f64 = 1.54578964390598;
    const REF_SD0: f64 = 24.7406579949841; // (Intercept) sd
    const REF_SD1: f64 = 5.92213765889808; // Days sd
    const REF_CORR: f64 = 0.0655512382381282;
    const REF_SIGMA: f64 = 25.5917957216753; // residual sd, lme4 sigma()

    let (x, y, n, p, model, ids) = sleepstudy_slope_design();

    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "sleepstudy slope LMM must converge");
    assert!(
        (f.beta[0] - REF_B0).abs() / REF_B0 < 1e-3,
        "β0 {} vs {REF_B0}",
        f.beta[0]
    );
    assert!(
        (f.beta[1] - REF_B1).abs() / REF_B1 < 1e-3,
        "β1 {} vs {REF_B1}",
        f.beta[1]
    );
    // Same se_rel band as grid/tol.R's cross-engine calibration (1e-3); measured
    // worst on this golden is 5.9e-6 (se0) / 5.0e-7 (se1), far inside it.
    assert!(
        (f.se[0] - REF_SE0).abs() / REF_SE0 < 1e-3,
        "se0 {} vs {REF_SE0}",
        f.se[0]
    );
    assert!(
        (f.se[1] - REF_SE1).abs() / REF_SE1 < 1e-3,
        "se1 {} vs {REF_SE1}",
        f.se[1]
    );
    // Gaussian dispersion = REML σ̂²; σ̂ against the frozen lme4 sigma().
    assert!(
        (f.dispersion.sqrt() - REF_SIGMA).abs() / REF_SIGMA < 1e-3,
        "σ̂ {} vs {REF_SIGMA}",
        f.dispersion.sqrt()
    );

    // Reference D (col-major vech lower-tri): [D00, D10, D11].
    let d00 = REF_SD0 * REF_SD0;
    let d11 = REF_SD1 * REF_SD1;
    let d10 = REF_CORR * REF_SD0 * REF_SD1;
    assert_eq!(f.varcorr.len(), 1, "one grouping block");
    let vc = &f.varcorr[0];
    assert_eq!(vc.len(), 3, "q=2 vech has 3 entries");
    assert!(
        (vc[0].sqrt() - REF_SD0).abs() / REF_SD0 < 1e-2,
        "sd0 {} vs {REF_SD0}",
        vc[0].sqrt()
    );
    assert!(
        (vc[2].sqrt() - REF_SD1).abs() / REF_SD1 < 1e-2,
        "sd1 {} vs {REF_SD1}",
        vc[2].sqrt()
    );
    assert!((vc[0] - d00).abs() / d00 < 1e-3, "D00 {} vs {d00}", vc[0]);
    assert!((vc[2] - d11).abs() / d11 < 1e-3, "D11 {} vs {d11}", vc[2]);
    // The off-diagonal covariance is the least-constrained θ coordinate under
    // BOBYQA's rho_end floor — θ10 is small relative to θ00/θ11, so it lands
    // with less relative precision than either variance. Measured against this
    // reference: D10 1.8e-5 relative, against 4.7e-7 on the Days stddev, so the
    // effect is real but worth about a factor of 40 — not the factor of 1e5 the
    // previous absolute-scale band (0.20·sd0·sd1) encoded. 1e-3 relative is the
    // cross-engine varcomp band this file's other glmm↔lme4 claims use.
    // The weighted analog (`fit_lmm_weighted_matches_lme4`) hits the same floor
    // on the same coordinate and states the band the same way — change together.
    assert!(
        (vc[1] - d10).abs() / d10.abs() < 1e-3,
        "D10 {} vs {d10}",
        vc[1]
    );

    assert!(f.n_eval > 0, "BOBYQA ran, evals must be counted");
    assert!(f.deviance.is_finite());
    assert!(!f.singular(), "sleepstudy is an interior optimum");
    let df = (n - p) as f64;
    let lme4_loglik = -871.814135979976; // validation/goldens/sleepstudy_lmm.json .estimates.loglik
    let remlcrit = -2.0 * lme4_loglik;
    let expected = remlcrit - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (f.deviance - expected).abs() < 1e-6,
        "deviance {} vs lme4-derived {expected}",
        f.deviance
    );
    // Fit.loglik must invert that stripped constant back to lme4's logLik —
    // the REML criterion on the logLik scale (reml flags it as such).
    assert!(
        (f.loglik - lme4_loglik).abs() < 1e-6,
        "loglik {} vs lme4 {lme4_loglik}",
        f.loglik
    );
    assert!(f.reml, "Gaussian LMM loglik is the REML criterion");
    assert_eq!(f.df, 6); // 2 β + 3 θ (q=2 vech) + σ²
}

/// Dense LMM with a per-row offset — the identity-link `y − o` shift — vs R
/// `lmer(offset=)`: sleepstudy random-slope with `o_i = 5·((i−1) mod 4)`
/// (0-based CSV row order in Rust). Oracle (R 4.5.3, lme4 1.1-38):
///   fl <- lmer(Reaction ~ Days + (Days | Subject), data = ss, offset = ol)
///   print(fixef(fl), digits = 15); print(REMLcrit(fl), digits = 15)
///   print(logLik(fl), digits = 15)
#[test]
fn fit_lmm_offset_matches_lme4() {
    const REF_BETA: [f64; 2] = [244.5869230303025, 10.3157708080802];
    const REF_REMLCRIT: f64 = 1756.8758930064;
    const REF_LOGLIK: f64 = -878.437946503201;
    let (x, y, n, p, model, ids) = sleepstudy_slope_design();
    let o: Vec<f64> = (0..n).map(|i| 5.0 * (i % 4) as f64).collect();

    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            offset: Some(o),
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "offset LMM must converge");
    for (j, (&b, &r)) in f.beta.iter().zip(&REF_BETA).enumerate() {
        assert!((b - r).abs() / r.abs() < 1e-3, "β[{j}] = {b} vs lme4 {r}");
    }
    let df = (n - p) as f64;
    let expected = REF_REMLCRIT - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (f.deviance - expected).abs() < 1e-6,
        "deviance {} vs lme4-derived {expected}",
        f.deviance
    );
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-6,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
}

/// Task 5: weighted dense LMM REML — sleepstudy random-slope fit with
/// synthetic weights `w_i = 1 + (i mod 3)` (i = 0-based CSV row order),
/// gated against a frozen lme4 golden. Pins β, SE, the 2×2 RE covariance
/// (SDs + correlation), σ̂, and the `−Σlog wᵢ` deviance-constant convention:
/// weighted REMLcrit strips the same `df·(1+ln 2π)` constant as the
/// unweighted case (see `lmm::reml_deviance`) PLUS the weighted Gaussian log-density's
/// `+½Σlog wᵢ` per row (`−Σlog wᵢ` on the −2ℓ deviance scale). Generated
/// with (R 4.5.3, lme4 1.1-38):
/// ```r
/// library(lme4)
/// d <- read.csv("validation/data/empirical/sleepstudy.csv")
/// w <- 1 + (seq_len(nrow(d)) - 1) %% 3
/// f <- lmer(Reaction ~ Days + (Days | Subject), data = d, weights = w, REML = TRUE)
/// print(summary(f)$coefficients, digits = 15)
/// print(as.data.frame(VarCorr(f)), digits = 15)
/// print(sigma(f), digits = 15); print(REMLcrit(f), digits = 15)
/// ```
#[test]
fn fit_lmm_weighted_matches_lme4() {
    const REF_B0: f64 = 251.804_690_405_274;
    const REF_B1: f64 = 10.4358707468765;
    const REF_SE0: f64 = 6.44698545564581;
    const REF_SE1: f64 = 1.57363056312657;
    const REF_SD0: f64 = 22.09852363841438; // (Intercept) sd
    const REF_SD1: f64 = 5.95218759898762; // Days sd
    const REF_CORR: f64 = 0.16395038320169;
    const REF_SIGMA: f64 = 38.62892535113247;
    const REF_REMLCRIT: f64 = 1778.29146275691;

    let (x, y, n, p, model, ids) = sleepstudy_slope_design();
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();

    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w.clone()),
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "weighted sleepstudy slope LMM must converge");
    assert!(
        (f.beta[0] - REF_B0).abs() / REF_B0 < 1e-6,
        "β0 {} vs {REF_B0}",
        f.beta[0]
    );
    assert!(
        (f.beta[1] - REF_B1).abs() / REF_B1 < 1e-6,
        "β1 {} vs {REF_B1}",
        f.beta[1]
    );
    assert!(
        (f.se[0] - REF_SE0).abs() / REF_SE0 < 1e-4,
        "se0 {} vs {REF_SE0}",
        f.se[0]
    );
    assert!(
        (f.se[1] - REF_SE1).abs() / REF_SE1 < 1e-4,
        "se1 {} vs {REF_SE1}",
        f.se[1]
    );

    assert_eq!(f.varcorr.len(), 1, "one grouping block");
    let vc = &f.varcorr[0];
    assert_eq!(vc.len(), 3, "q=2 vech has 3 entries");
    let sd0 = vc[0].sqrt();
    let sd1 = vc[2].sqrt();
    let corr = vc[1] / (sd0 * sd1);
    assert!(
        (sd0 - REF_SD0).abs() / REF_SD0 < 1e-4,
        "sd0 {sd0} vs {REF_SD0}"
    );
    assert!(
        (sd1 - REF_SD1).abs() / REF_SD1 < 1e-4,
        "sd1 {sd1} vs {REF_SD1}"
    );
    // The off-diagonal correlation is the least-constrained θ coordinate under
    // BOBYQA's rho_end floor (θ10 is small relative to θ00/θ11, so its relative
    // precision is looser) — the unweighted analog
    // (`fit_sleepstudy_slope_varcorr_matches_lme4`) hits the same floor on the
    // same coordinate and states its band the same way; change together.
    // Absolute rather than relative because a correlation near zero has no
    // meaningful relative scale. Measured gap against this reference: 3.0e-5.
    assert!((corr - REF_CORR).abs() < 4e-3, "corr {corr} vs {REF_CORR}");

    // Fit.deviance vs REMLcrit(f) − (n−p)·(1+ln 2π) — pins the −Σlog wᵢ
    // constant `fit_mle` folds into the reported deviance (see the arm
    // above fit_mle in this file). 1e-6 abs, as the unweighted analog above.
    let df = (n - p) as f64;
    let expected = REF_REMLCRIT - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (f.deviance - expected).abs() < 1e-6,
        "deviance {} vs lme4-derived {expected}",
        f.deviance
    );
    // loglik = −REMLcrit/2 under weights — pins that the −Σlog wᵢ correction
    // lands INSIDE the criterion the loglik reports (lme4's weighted logLik).
    assert!(
        (f.loglik - (-REF_REMLCRIT / 2.0)).abs() < 1e-6,
        "weighted loglik {} vs lme4 {}",
        f.loglik,
        -REF_REMLCRIT / 2.0
    );
    assert!(f.reml);

    // σ̂ isn't exposed on `Fit` for q≥2 RE (tau2 only reproduces the (0,0)
    // diagonal, not the raw residual variance) — reconstruct via the same
    // suff-stats accumulator/kernel `fit_mle` calls, reading `sigma_sq`
    // straight off `LmmFit` (mirrors fit_mle's construction verbatim).
    let (sized, ids, _perm) = spec_sized_from_ids(&model, &ids);
    let mut ws = LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &[1], &[], false);
    let mut x_mat = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }
    ws.suff_mut()
        .add_rows_multi(x_mat.as_ref(), &y, &ids.primary, &[], Some(&w));
    let lmm_fit = fit_lmm_unhinted(&mut ws, &[0, 1], None, ColumnScaling::IDENTITY);
    let sigma = lmm_fit.sigma_sq.sqrt();
    assert!(
        (sigma - REF_SIGMA).abs() / REF_SIGMA < 1e-4,
        "sigma {sigma} vs {REF_SIGMA}"
    );
}

/// Varying prior weights on a CROSSED random-slope design
/// (`y ~ 1 + x + (1 + x | g1) + (1 | g2)`, the `sim_slope` fixture): the dense
/// kernel must give the sparse kernel's β/SE/varcorr. A q_p = 2 primary slope
/// plus a crossed intercept extra takes the dense kernel's scalar crossed
/// branch, whose weight sites in `add_rows_multi` are the intercept×intercept
/// `zx += wᵢ` and the slope↔crossed `zx_slope += z·zw`. Unit weights cannot
/// tell those from a wrong-power bug, and a constant weight is rescaled away
/// before the kernel runs (`accumulate_lmm_rows` divides by
/// `family::weight_scale`), so the weights vary by row. The dense side is
/// forced through `fit_mle_noz_pub`, as in
/// `fit_lmm_crossed_slope_extra_varying_weights_noz_matches_sparse`.
#[test]
fn fit_lmm_scalar_crossed_varying_weights_noz_matches_sparse() {
    let csv = include_str!("../../validation/data/simulated/sim_slope.csv");
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut g1_raw = Vec::<String>::new();
    let mut g2_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // y
        xcol.push(f[1].parse().unwrap()); // x
        g1_raw.push(f[2].to_string());
        g2_raw.push(f[3].to_string());
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let (g1, _n1) = dense_str(&g1_raw);
    let (g2, _n2) = dense_str(&g2_raw);
    // Varying, O(1), strictly positive (`lcg` ranges over [-1, 1]).
    let mut st = 1301u64;
    let w: Vec<f64> = (0..n).map(|_| 1.5 + 1.3 * lcg(&mut st)).collect();

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![1], // random slope on x for g1
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![], // g2 intercept-only
            }],
        }),
    };
    let ids = GroupIds {
        primary: g1,
        extra: vec![g2],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        ..FitOptions::default()
    };
    let (sized, ids, _perm) = spec_sized_from_ids(&model, &ids);
    let noz = fit_mle_noz_pub(&x, &y, n, p, &sized, &ids.primary, &ids.extra, None, &opts);
    let sp = fit_mle_sparse_pub(&x, &y, n, p, &sized, &ids.primary, &ids.extra, None, &opts);
    assert!(
        noz.converged() && sp.converged(),
        "both routes must converge"
    );
    // Two independent BOBYQA solves, bounded by the solver's rho_end floor.
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    for j in 0..p {
        assert!(
            rel(sp.beta[j], noz.beta[j]) < 1e-4,
            "β[{j}] sparse={} noz={}",
            sp.beta[j],
            noz.beta[j]
        );
        assert!(
            rel(sp.se[j], noz.se[j]) < 1e-4,
            "se[{j}] sparse={} noz={}",
            sp.se[j],
            noz.se[j]
        );
    }
    // varcorr covers BOTH groupings' D̂ blocks (tau2 only reproduces the
    // (0,0) diagonal for the q=2 primary).
    assert_eq!(sp.varcorr.len(), noz.varcorr.len(), "varcorr block count");
    for (bi, (sb, nb)) in sp.varcorr.iter().zip(noz.varcorr.iter()).enumerate() {
        assert_eq!(sb.len(), nb.len(), "varcorr[{bi}] len");
        for (ei, (a, b)) in sb.iter().zip(nb.iter()).enumerate() {
            assert!(rel(*a, *b) < 1e-4, "varcorr[{bi}][{ei}] sparse={a} noz={b}");
        }
    }
}

/// The test-only `fit_mle` baseline fits the criterion `opts.reml` resolves
/// to, on both kernels: its σ̂² is the one `fit_cold` reports under the same
/// option, so a forced-route parity test under ML compares ML with ML.
#[test]
fn fit_mle_baseline_honours_reml() {
    let (x, y, n, p) = super::common_tests::lmm_hand_dataset();
    let model = intercept_only_spec(Sizing::FixedClusters { n_clusters: 6 });
    let ids = GroupIds {
        primary: (0..n).map(|i| (i % 6) as u32).collect(),
        extra: vec![],
    };
    let (sized, ids, _perm) = spec_sized_from_ids(&model, &ids);
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    for reml in [None, Some(false)] {
        let opts = FitOptions {
            target_indices: vec![1, 2],
            reml,
            ..FitOptions::default()
        };
        let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
        let noz = fit_mle_noz_pub(&x, &y, n, p, &sized, &ids.primary, &ids.extra, None, &opts);
        let sp = fit_mle_sparse_pub(&x, &y, n, p, &sized, &ids.primary, &ids.extra, None, &opts);
        for (route, fit) in [("noz", noz), ("sparse", sp)] {
            assert!(
                rel(fit.dispersion, cold.dispersion) < 1e-5,
                "reml = {reml:?}, {route}: σ̂² {} vs fit_cold {}",
                fit.dispersion,
                cold.dispersion
            );
        }
    }
}

/// `y ~ 1 + x1 + (1 | g)` whose ML variance pins at exactly zero: within each
/// cluster the residuals alternate ±0.8 and cancel (the design of
/// `fit_lmm_weighted_boundary_matches_wls`), and `x1` is on a scale of 100 so
/// the column standardization is far from the identity.
fn ml_pinned_design() -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<f64>, usize, usize) {
    let (n, n_clusters, p) = (48usize, 6usize, 2usize);
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = 100.0 * lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
        y[i] = 0.5 + 0.004 * x1 + e;
        w[i] = 1.0 + (i % 3) as f64;
    }
    (x, y, ids, w, n, p)
}

/// An ML Gaussian mixed fit pinned at θ̂ = 0 is the fixed-effects model, so its
/// log-likelihood is the OLS (ML) log-likelihood on the same X: `ln|L_ZZ|² = 0`,
/// `r²` is the (weighted) RSS and `ν = n`. A REML standardization shift, the
/// REML `ν = n − p`, or a weighted correction on the wrong scale each move it by
/// O(1). The ML fit reports `reml == false` and the REML fit's `df`.
#[test]
fn ml_fit_pinned_at_zero_has_the_ols_loglik() {
    let (x, y, ids, w, n, p) = ml_pinned_design();
    let ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let fixed_only = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    for weights in [None, Some(w)] {
        let tag = if weights.is_some() {
            "weighted"
        } else {
            "plain"
        };
        let opts = |reml: Option<bool>| FitOptions {
            target_indices: vec![0, 1],
            weights: weights.clone(),
            reml,
            ..FitOptions::default()
        };
        let ml = fit_cold(
            &x,
            &y,
            n,
            p,
            &intercept_only_lmm(),
            &ids,
            &opts(Some(false)),
        );
        let reml = fit_cold(&x, &y, n, p, &intercept_only_lmm(), &ids, &opts(None));
        let ols = fit_cold(&x, &y, n, p, &fixed_only, &GroupIds::default(), &opts(None));
        assert!(ml.converged() && ols.converged(), "{tag}: both fit");
        assert_eq!(ml.tau2, vec![0.0], "{tag}: the ML fit must pin at θ̂ = 0");
        assert!(
            (ml.loglik - ols.loglik).abs() <= 1e-10 * ols.loglik.abs(),
            "{tag}: ML logLik {} vs OLS {}",
            ml.loglik,
            ols.loglik
        );
        // σ̂²_ML = RSS_w/n: the OLS dispersion RSS_w/(n − p) times (n − p)/n,
        // and each SE the OLS one times √((n − p)/n).
        let shrink = (n - p) as f64 / n as f64;
        assert!(
            (ml.dispersion - ols.dispersion * shrink).abs() <= 1e-10 * ols.dispersion,
            "{tag}: ML dispersion {} vs OLS {} · (n − p)/n",
            ml.dispersion,
            ols.dispersion
        );
        for j in 0..p {
            assert!(
                (ml.se[j] - ols.se[j] * shrink.sqrt()).abs() <= 1e-10 * ols.se[j],
                "{tag}: ML se[{j}] {} vs OLS {} · √((n − p)/n)",
                ml.se[j],
                ols.se[j]
            );
        }
        assert!(!ml.reml && reml.reml, "{tag}: reml flags");
        assert_eq!(ml.df, p + 2, "{tag}: ML df");
        assert_eq!(ml.df, reml.df, "{tag}: df is the same on both criteria");
    }
}

/// The ML fit's deviance is the ML criterion's minimum: no larger than the ML
/// deviance at the REML fit's θ̂. On this design (group-level covariate, 8
/// groups) the two θ̂ differ by far more than `lmm::RHO_END`, and `x1` on a
/// scale of 100 gives the column standardization a large REML shift, which
/// the ML deviance must not carry. The ML deviance at a θ is evaluated on a
/// dense kernel over the caller's X (ML is invariant to the standardization),
/// and first checked against the fit's own deviance at its θ̂.
#[test]
fn ml_deviance_at_ml_theta_is_below_ml_deviance_at_reml_theta() {
    let (a, m, p) = (8usize, 5usize, 3usize);
    let n = a * m;
    let mut st = 61u64;
    let u: Vec<f64> = (0..a).map(|_| 0.6 * lcg(&mut st)).collect();
    let zg: Vec<f64> = (0..a).map(|_| lcg(&mut st)).collect();
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let g = i % a;
        ids[i] = g as u32;
        let x1 = 100.0 * lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = zg[g];
        y[i] = 1.0 + 0.003 * x1 + 0.5 * zg[g] + u[g] + 0.5 * lcg(&mut st);
    }
    let ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let opts = |reml: Option<bool>| FitOptions {
        target_indices: (0..p as u32).collect(),
        reml,
        ..FitOptions::default()
    };
    let ml = fit_cold(
        &x,
        &y,
        n,
        p,
        &intercept_only_lmm(),
        &ids,
        &opts(Some(false)),
    );
    let reml = fit_cold(&x, &y, n, p, &intercept_only_lmm(), &ids, &opts(None));
    assert!(ml.converged() && reml.converged(), "both fit");
    assert!(!ml.singular() && !reml.singular(), "both optima interior");
    assert!(!ml.reml && reml.reml, "reml flags");
    assert_eq!(ml.df, reml.df, "df is the same on both criteria");
    let theta = |f: &Fit| (f.tau2[0] / f.dispersion).sqrt();
    let (theta_ml, theta_reml) = (theta(&ml), theta(&reml));
    assert!(
        (theta_reml - theta_ml).abs() > 1e3 * crate::lmm::RHO_END * theta_reml,
        "θ̂_ML {theta_ml} and θ̂_REML {theta_reml} must differ beyond the optimizer's accuracy"
    );

    let mut ws = LmmWorkspace::new(p, a);
    ws.fit_mut().criterion = crate::lmm::Criterion::Ml;
    let x_mat = Mat::<f64>::from_fn(n, p, |i, j| x[i * p + j]);
    ws.suff_mut().add_rows(x_mat.as_ref(), &y, &ids.primary);
    let at_ml = ws.kernel.deviance(&[theta_ml]);
    let at_reml = ws.kernel.deviance(&[theta_reml]);
    assert!(
        (ml.deviance - at_ml).abs() <= 1e-8 * at_ml.abs(),
        "the ML fit's deviance {} vs the ML criterion at its θ̂ {at_ml}",
        ml.deviance
    );
    assert!(
        ml.deviance <= at_reml,
        "ML deviance at θ̂_ML {} above the ML deviance at θ̂_REML {at_reml}",
        ml.deviance
    );
}

/// Varying prior weights on a design where the EXTRA (crossed) grouping
/// itself carries a random slope: `y ~ 1 + x1 + (1+x1 | primary) + (1+x1 |
/// extra)`. `extra_slopes_any` is true, so the dense kernel's `zx` fill takes
/// the family-blocked path (`kernel.rs`'s "Blocked crossed/nested-slopes
/// path"), not the scalar `zx`/`zx_slope` branch the other crossed weighted
/// tests exercise. A per-row weight applied to the wrong factor in that fill
/// would move the dense fit off the sparse kernel's answer; a constant weight
/// cannot show that, since w≡c only rescales θ̂ uniformly. `classify_design`
/// always routes `extra_slopes_any` to Sparse, so the dense side is forced
/// through `fit_mle_noz_pub`, the same forced-NoZ entry the grid cross-checks
/// in `src/sparse/tests.rs` use.
#[test]
fn fit_lmm_crossed_slope_extra_varying_weights_noz_matches_sparse() {
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 5 },
            slopes: vec![1],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                slopes: vec![1],
            }],
        }),
    };
    let n = 60;
    let mut st = 91u64;
    let u0p: Vec<f64> = (0..5).map(|_| 0.5 * lcg(&mut st)).collect();
    let u1p: Vec<f64> = (0..5).map(|_| 0.3 * lcg(&mut st)).collect();
    let u0e: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
    let u1e: Vec<f64> = (0..4).map(|_| 0.3 * lcg(&mut st)).collect();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        let par = cluster_of_row(&cluster.re.as_ref().unwrap().sizing, i);
        let item = extra_level_of_row(&cluster, 0, i) as usize;
        pid[i] = par as u32;
        eid[i] = item as u32;
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        y[i] = 0.5
            + 0.4 * x1
            + u0p[par]
            + u1p[par] * x1
            + u0e[item]
            + u1e[item] * x1
            + 0.8 * lcg(&mut st);
        // Varies by row — a weight folded into the wrong RE column shows up
        // as a β/varcorr mismatch here, where a constant factor would only
        // rescale both routes alike.
        w[i] = 1.0 + (i % 3) as f64;
    }
    let ids = GroupIds {
        primary: pid,
        extra: vec![eid],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        ..FitOptions::default()
    };
    let (sized, ids, _perm) = spec_sized_from_ids(&cluster, &ids);
    let noz = fit_mle_noz_pub(&x, &y, n, p, &sized, &ids.primary, &ids.extra, None, &opts);
    let sp = fit_mle_sparse_pub(&x, &y, n, p, &sized, &ids.primary, &ids.extra, None, &opts);
    assert!(
        noz.converged() && sp.converged(),
        "both routes must converge"
    );
    // Two independent BOBYQA solves, not a shared trajectory — bounded by the
    // solver's own rho_end floor, same order as the unweighted grid
    // cross-check in `src/sparse/tests.rs` (`run_grid_agreement`'s 1e-4).
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    for j in 0..p {
        assert!(
            rel(sp.beta[j], noz.beta[j]) < 1e-4,
            "β[{j}] sparse={} noz={}",
            sp.beta[j],
            noz.beta[j]
        );
        assert!(
            rel(sp.se[j], noz.se[j]) < 1e-4,
            "se[{j}] sparse={} noz={}",
            sp.se[j],
            noz.se[j]
        );
    }
    assert_eq!(sp.varcorr.len(), noz.varcorr.len(), "varcorr block count");
    for (bi, (sb, nb)) in sp.varcorr.iter().zip(noz.varcorr.iter()).enumerate() {
        assert_eq!(sb.len(), nb.len(), "varcorr[{bi}] len");
        for (ei, (a, b)) in sb.iter().zip(nb.iter()).enumerate() {
            assert!(rel(*a, *b) < 1e-4, "varcorr[{bi}][{ei}] sparse={a} noz={b}");
        }
    }
}

/// Weight-SCALE invariance: varying base weights
/// `w`, rescaled by `c ∈ {2^20, 2^-20, 1e8, 1e-8}`, must reproduce the `c = 1`
/// fit's β, SE and τ² (both routes forced via `fit_mle_noz_pub`/
/// `fit_mle_sparse_pub`, the same forced entries the crossed-slope test above
/// uses), move `dispersion` by exactly `c` (up to solver noise), and leave
/// `loglik` unchanged. BOBYQA searches `ŵ = w/family::weight_scale(w)`, not
/// raw `w`: searching raw `w` would move the internal θ at `c = 2^20`/`2^-20`
/// far enough to hit `THETA_HI`/`PIN_THETA` (`lmm::mod::THETA_HI`,
/// `PIN_THETA`) and both SE and dispersion would come out wrong; `2^20`/`2^-20`
/// are exact powers of two, so `ŵ` is bit-identical to the `c = 1` case and the
/// tolerance can be tight (1e-6).
/// `1e8`/`1e-8` are not powers of two, so `s` differs by one ULP-scale factor
/// from a clean rescale and gets a looser band (still far tighter than a
/// raw-`w` search's error, which would be off by orders of magnitude).
#[test]
fn fit_lmm_weight_scale_invariant_dense_and_sparse() {
    let n_clusters = 8usize;
    let per = 6usize;
    let n = n_clusters * per;
    let p = 2usize;
    let mut st = 409u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids_v = vec![0u32; n];
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        ids_v[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        let re = 0.3 * ((ids_v[i] as f64) - (n_clusters as f64) / 2.0);
        y[i] = 0.5 + 0.4 * x1 + re + 0.2 * lcg(&mut st);
        // Varying, O(1), strictly positive base weights (`lcg` ranges over
        // [-1, 1]) — a constant weight cannot show the per-row `ŵ` division
        // landing on the wrong factor.
        w[i] = 1.5 + 1.3 * lcg(&mut st);
    }
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: ids_v,
        extra: vec![],
    };
    let (sized, ids, _perm) = spec_sized_from_ids(&cluster, &ids);

    let fit_at = |c: f64, reml: Option<bool>| {
        let opts = FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w.iter().map(|&wi| wi * c).collect()),
            reml,
            ..FitOptions::default()
        };
        let dense = fit_mle_noz_pub(&x, &y, n, p, &sized, &ids.primary, &ids.extra, None, &opts);
        let sparse =
            fit_mle_sparse_pub(&x, &y, n, p, &sized, &ids.primary, &ids.extra, None, &opts);
        (dense, sparse)
    };

    // Both criteria: the weighted correction and the scale cancellation are
    // the same argument under REML and ML.
    for reml in [None, Some(false)] {
        let (base_dense, base_sparse) = fit_at(1.0, reml);
        assert!(
            base_dense.converged() && base_sparse.converged(),
            "{reml:?} c=1 baseline must converge (dense={} sparse={})",
            base_dense.converged(),
            base_sparse.converged()
        );

        let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
        for &(c, tol) in &[
            (2f64.powi(20), 1e-6),
            (2f64.powi(-20), 1e-6),
            (1e8, 1e-4),
            (1e-8, 1e-4),
        ] {
            let (dense, sparse) = fit_at(c, reml);
            for (name, base, scaled) in [
                ("dense", &base_dense, &dense),
                ("sparse", &base_sparse, &sparse),
            ] {
                assert!(scaled.converged(), "{name} {reml:?} c={c} must converge");
                for j in 0..p {
                    assert!(
                        rel(scaled.beta[j], base.beta[j]) < tol,
                        "{name} {reml:?} c={c} β[{j}] {} vs base {}",
                        scaled.beta[j],
                        base.beta[j]
                    );
                    assert!(
                        rel(scaled.se[j], base.se[j]) < tol,
                        "{name} {reml:?} c={c} se[{j}] {} vs base {}",
                        scaled.se[j],
                        base.se[j]
                    );
                }
                assert_eq!(scaled.tau2.len(), base.tau2.len());
                for k in 0..base.tau2.len() {
                    assert!(
                        rel(scaled.tau2[k], base.tau2[k]) < tol,
                        "{name} {reml:?} c={c} tau2[{k}] {} vs base {}",
                        scaled.tau2[k],
                        base.tau2[k]
                    );
                }
                // σ̂² is NOT invariant: it moves with the weight scale itself.
                assert!(
                    rel(scaled.dispersion, c * base.dispersion) < tol,
                    "{name} {reml:?} c={c} dispersion {} vs c*base {}",
                    scaled.dispersion,
                    c * base.dispersion
                );
                assert!(
                    rel(scaled.loglik, base.loglik) < tol,
                    "{name} {reml:?} c={c} loglik {} vs base {}",
                    scaled.loglik,
                    base.loglik
                );
            }
        }
    }
}

/// The dense-LMM boundary (τ̂ ≈ 0, pinned exactly — mirrors
/// `lmm::tests::zero_between_cluster_variance_pins_at_exactly_zero`) must
/// reproduce the weighted fixed-only WLS fit on the
/// same rows: at θ̂=0 the mixed kernel's weighted Grams (`c`/`s`/`counts`,
/// all Σwᵢ-scaled) collapse to the same weighted
/// normal equations WLS solves directly, so the two paths must agree.
#[test]
fn fit_lmm_weighted_boundary_matches_wls() {
    let n = 48usize;
    let n_clusters = 6usize;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        // i/n_clusters cycles 0..8 within each cluster: 4 even, 4 odd ⇒
        // the ±0.8 residuals cancel exactly per cluster (deterministic pin).
        let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
        y[i] = 0.5 + 0.4 * x1 + e;
        w[i] = 1.0 + (i % 3) as f64;
    }

    let mixed_model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let mixed_ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let mixed = fit_cold(
        &x,
        &y,
        n,
        2,
        &mixed_model,
        &mixed_ids,
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w.clone()),
            ..FitOptions::default()
        },
    );
    assert!(mixed.converged(), "boundary pin still counts as converged");
    assert!(mixed.singular(), "must pin at the τ=0 boundary");

    let fixed_only = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let wls = fit_cold(
        &x,
        &y,
        n,
        2,
        &fixed_only,
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w),
            ..FitOptions::default()
        },
    );
    assert!(wls.converged());

    for j in 0..2 {
        assert!(
            (mixed.beta[j] - wls.beta[j]).abs() / wls.beta[j].abs() < 1e-6,
            "β[{j}] mixed {} vs WLS {}",
            mixed.beta[j],
            wls.beta[j]
        );
        assert!(
            (mixed.se[j] - wls.se[j]).abs() / wls.se[j] < 1e-3,
            "se[{j}] mixed {} vs WLS {}",
            mixed.se[j],
            wls.se[j]
        );
    }
}

/// Crossed random-slope `y ~ 1 + x + (1 + x | g1) + (1 | g2)`: a q=2 `varcorr`
/// block on the PRIMARY plus a scalar block on a crossed EXTRA grouping — the
/// multi-grouping generalization the single-grouping composition omits. A
/// permuted or mis-sized varcorr layout moves these numbers.
///
/// Values recorded from glmm. They are validated by `sim_slope_lmm`, whose
/// cross-engine cell checks the same fit against lme4.
///
/// Re-pinned 2026-08-23 with random-effect design column scaling: `x` has a
/// column scale of 0.9688, so this fit carries a real scale factor and sits in
/// the reassociation band the change allows (worst move here 1.3e-6 on the
/// `g1` covariance, on a criterion that improved by 4.9e-8).
#[test]
fn fit_sim_slope_varcorr_is_pinned() {
    // Re-pinned 2026-10-01: the design is standardized internally before the fit (x̃ = (x − m)/s, results mapped back).
    // The REML criterion is unchanged to 1e-12 (27.615443411970); the θ search
    // stops at another point of a flat direction: g1's covariance moved 1.6e-6
    // relative, its variances 3.3e-7 and 1.4e-7, σ̂² 8.2e-8, g2 5.5e-8, the SEs
    // 7.8e-8, β 5.7e-10.
    // Re-pinned 2026-10-02 with the per-search BOBYQA start radii (LMM blind
    // start 0.5): the search takes another path to the same optimum, worst move
    // here 1.4e-6 on the g1 covariance (variances 5.8e-7 and 4.5e-7, σ̂² 1.9e-7,
    // g2 1.6e-7, the SEs 1.5e-7, β 3.8e-9), REML criterion 1.1e-13 lower, no
    // converged, singular or pinned flag changed.
    const REF_BETA: [f64; 2] = [1.0380272315443226, 0.8009679255408596];
    const REF_SE: [f64; 2] = [0.33893198001615327, 0.17067326068596858];
    // varcorr[0] = g1, packed lower triangle [v00, c01, v11]; varcorr[1] = g2.
    const REF_VC_G1: [f64; 3] = [
        0.8994004671804791,
        -0.11776411602334781,
        0.39740690350927466,
    ];
    const REF_VC_G2: f64 = 0.508186655377839;
    const REF_SIGMA2: f64 = 0.5717628048921557;

    let csv = include_str!("../../validation/data/simulated/sim_slope.csv");
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut g1_raw = Vec::<String>::new();
    let mut g2_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // y
        xcol.push(f[1].parse().unwrap()); // x
        g1_raw.push(f[2].to_string());
        g2_raw.push(f[3].to_string());
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let (g1, _n1) = dense_str(&g1_raw);
    let (g2, _n2) = dense_str(&g2_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![1], // random slope on x for g1
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![], // g2 intercept-only
            }],
        }),
    };
    let ids = GroupIds {
        primary: g1,
        extra: vec![g2],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );

    assert!(f.converged());
    assert_pinned(&f.beta, &REF_BETA, PIN_REL_ITER, "beta");
    assert_pinned(&f.se, &REF_SE, PIN_REL_ITER, "se");
    assert_eq!(f.varcorr.len(), 2, "one block per grouping, g1 then g2");
    assert_pinned(&f.varcorr[0], &REF_VC_G1, PIN_REL_ITER, "g1 varcorr");
    assert_pinned(&f.varcorr[1], &[REF_VC_G2], PIN_REL_ITER, "g2 varcorr");
    assert_pinned(&[f.dispersion], &[REF_SIGMA2], PIN_REL_ITER, "sigma2");
}

/// Gap #1 crossed: Penicillin `diameter ~ 1 + (1|plate) + (1|sample)` through the
/// data-shaped `fit_cold` with `GroupIds { primary: plate, extra: vec![sample] }`,
/// gated against the frozen lme4 golden (`validation/goldens/penicillin_lmm.json`,
/// REML). Two crossed intercept-only groupings, fixed effect = intercept only
/// (p=1). Placeholder spec counts prove the data path derives level counts from
/// the ids. The oracle is sacred.
#[test]
fn fit_penicillin_crossed_matches_lme4() {
    const REF_BETA: f64 = 22.9722222222;
    const REF_SE: f64 = 0.808595361386;
    const REF_PLATE_SD: f64 = 0.846702;
    const REF_SAMPLE_SD: f64 = 1.931614;

    let csv = include_str!("../../validation/data/empirical/Penicillin.csv");
    let mut y = Vec::<f64>::new();
    let mut plate_raw = Vec::<String>::new();
    let mut sample_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // diameter
        plate_raw.push(f[1].to_string());
        sample_raw.push(f[2].to_string());
    }
    let n = y.len();
    let p = 1;
    let x = vec![1.0f64; n]; // intercept-only design
    let (plate, _n_plate) = dense_str(&plate_raw);
    let (sample, _n_sample) = dense_str(&sample_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 }, // placeholder
                slopes: vec![],
            }],
        }),
    };
    let ids = GroupIds {
        primary: plate,
        extra: vec![sample],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0],
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "Penicillin crossed LMM must converge");
    assert!(
        (f.beta[0] - REF_BETA).abs() / REF_BETA < 1e-4,
        "β0 = {} vs lme4 {REF_BETA}",
        f.beta[0]
    );
    // Same se_rel band as grid/tol.R's cross-engine calibration (1e-3); measured
    // worst on this golden is 2.7e-5, far inside it.
    let se_rel = (f.se[0] - REF_SE).abs() / REF_SE;
    assert!(
        se_rel < 1e-3,
        "se0 = {} vs lme4 {REF_SE} (rel {se_rel})",
        f.se[0]
    );
    // theta layout: [primary (plate) vech | sample scalar]; tau2[k] = θ̂[k]²·σ̂².
    let plate_sd = f.tau2[0].sqrt();
    let sample_sd = f.tau2[1].sqrt();
    assert!(
        (plate_sd - REF_PLATE_SD).abs() / REF_PLATE_SD < 5e-3,
        "plate sd = {plate_sd} vs lme4 {REF_PLATE_SD}"
    );
    assert!(
        (sample_sd - REF_SAMPLE_SD).abs() / REF_SAMPLE_SD < 5e-3,
        "sample sd = {sample_sd} vs lme4 {REF_SAMPLE_SD}"
    );
}

/// Gap #1 nested: Pastes `strength ~ 1 + (1|batch/cask)` through the data-shaped
/// `fit_cold` with `GroupIds { primary: batch, extra: vec![cask] }`, where `cask`
/// is the globally-unique batch:cask level (dense 0..29). Gated against the frozen
/// lme4 golden (`validation/goldens/pastes_lmm.json`, REML). Exercises the
/// `NestedWithin` topology tag on the data path; placeholder counts prove level
/// counts come from the ids. The oracle is sacred.
#[test]
fn fit_pastes_nested_matches_lme4() {
    const REF_BETA: f64 = 60.0533333333;
    const REF_SE: f64 = 0.676870215074;
    const REF_BATCH_SD: f64 = 1.287366;
    const REF_CASK_SD: f64 = 2.904077;

    let csv = include_str!("../../validation/data/empirical/Pastes.csv");
    // cols: strength,batch,cask,sample  (sample = "batch:cask" global label)
    let mut y = Vec::<f64>::new();
    let mut batch_raw = Vec::<String>::new();
    let mut cask_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // strength
        batch_raw.push(f[1].to_string()); // batch
        cask_raw.push(f[3].to_string()); // sample = batch:cask global label
    }
    let n = y.len();
    let p = 1;
    let x = vec![1.0f64; n];
    let (batch, _n_batch) = dense_str(&batch_raw);
    let (cask, _n_cask) = dense_str(&cask_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 1 }, // placeholder
                slopes: vec![],
            }],
        }),
    };
    let ids = GroupIds {
        primary: batch,
        extra: vec![cask],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0],
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "Pastes nested LMM must converge");
    assert!(
        (f.beta[0] - REF_BETA).abs() / REF_BETA < 1e-4,
        "β0 = {} vs lme4 {REF_BETA}",
        f.beta[0]
    );
    // Same se_rel band as grid/tol.R's cross-engine calibration (1e-3); measured
    // worst on this golden is 2.4e-6, far inside it.
    let se_rel = (f.se[0] - REF_SE).abs() / REF_SE;
    assert!(
        se_rel < 1e-3,
        "se0 = {} vs lme4 {REF_SE} (rel {se_rel})",
        f.se[0]
    );
    // theta layout: [primary (batch) vech | nested (cask) scalar].
    let batch_sd = f.tau2[0].sqrt();
    let cask_sd = f.tau2[1].sqrt();
    assert!(
        (batch_sd - REF_BATCH_SD).abs() / REF_BATCH_SD < 1e-2,
        "batch sd = {batch_sd} vs lme4 {REF_BATCH_SD}"
    );
    assert!(
        (cask_sd - REF_CASK_SD).abs() / REF_CASK_SD < 5e-3,
        "cask sd = {cask_sd} vs lme4 {REF_CASK_SD}"
    );
}

/// Grouping order must not change the answer. Both declarations of the same
/// two-factor scalar-crossed design — `(1|big) + (1|small)` and
/// `(1|small) + (1|big)` — reach the kernel as the identical design, because
/// the size rule makes the many-level factor the primary either way. So the
/// fits are BIT-identical wherever the quantity is grouping-blind (deviance, β,
/// SE, the optimizer's evaluation count), and every grouping-indexed field
/// comes back in the order that caller declared, which is the reverse of the
/// other's.
///
/// Bit equality is the assertion, not a tolerance: the two calls hand the
/// accumulator the same rows in the same order, so anything short of identity
/// would mean the reorder had leaked into the arithmetic.
#[test]
fn scalar_crossed_lmm_is_grouping_order_insensitive() {
    const BIG: usize = 15;
    const SMALL: usize = 4;
    let n = 180usize;
    let p = 2usize;
    // Seed 1: with `small_eff` zeroed out below, this draw lands the small
    // grouping's REML variance estimate on the PIN_THETA boundary, so
    // `diagnostics.pinned` has a real block for the order-swap check below to
    // compare (most seeds do not pin; this one is kept for that reason).
    let mut st = 1u64;
    let big_eff: Vec<f64> = (0..BIG).map(|_| 0.9 * lcg(&mut st)).collect();
    // Zero true variance on the small grouping (unlike big_eff, which is
    // sampled) is what makes a boundary estimate possible at all.
    let small_eff: Vec<f64> = vec![0.0; SMALL];
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut big = vec![0u32; n];
    let mut small = vec![0u32; n];
    for i in 0..n {
        big[i] = (i % BIG) as u32;
        small[i] = ((i / BIG) % SMALL) as u32;
        let cov = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = cov;
        y[i] = 0.7
            + 0.4 * cov
            + big_eff[big[i] as usize]
            + small_eff[small[i] as usize]
            + 0.25 * lcg(&mut st);
    }
    // One extra crossed intercept-only grouping; counts are placeholders, as
    // the formula frontend emits them.
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![],
            }],
        }),
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let fit_of = |primary: &[u32], extra: &[u32]| {
        let ids = GroupIds {
            primary: primary.to_vec(),
            extra: vec![extra.to_vec()],
        };
        fit_cold(&x, &y, n, p, &model, &ids, &opts)
    };
    // `a` declares the many-level factor first (already the chosen order),
    // `b` declares it second (the rule swaps).
    let a = fit_of(&big, &small);
    let b = fit_of(&small, &big);
    assert!(a.converged() && b.converged(), "both orders must converge");

    assert_eq!(a.deviance.to_bits(), b.deviance.to_bits(), "deviance");
    assert_eq!(a.n_eval, b.n_eval, "objective evaluations");
    for j in 0..p {
        assert_eq!(a.beta[j].to_bits(), b.beta[j].to_bits(), "beta[{j}]");
        assert_eq!(a.se[j].to_bits(), b.se[j].to_bits(), "se[{j}]");
    }

    // Declared order: `a` reports [big, small], `b` reports [small, big].
    assert_eq!(a.ranef_levels, vec![BIG, SMALL], "a declares big first");
    assert_eq!(b.ranef_levels, vec![SMALL, BIG], "b declares small first");
    assert_eq!(a.varcorr.len(), 2, "one block per grouping");
    assert_eq!(b.varcorr.len(), 2, "one block per grouping");
    for (g, h) in [(0usize, 1usize), (1, 0)] {
        assert_eq!(
            a.varcorr[g].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.varcorr[h].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "varcorr block for the same grouping (a[{g}] vs b[{h}])"
        );
        assert_eq!(
            a.tau2[g].to_bits(),
            b.tau2[h].to_bits(),
            "tau2[{g}] vs tau2[{h}]"
        );
    }
    assert_eq!(
        a.diagnostics.pinned.len(),
        b.diagnostics.pinned.len(),
        "pinned block count"
    );
    // Not vacuous: this design (seed 1, zero true small-grouping variance)
    // pins the small grouping's diagonal, so both sides carry a real block —
    // an empty `pinned` here would mean the fixture stopped pinning and the
    // swap check below stopped checking anything.
    assert!(
        !a.diagnostics.pinned.is_empty(),
        "fixture must pin the small grouping's variance for this check to compare anything"
    );
    for (g, h) in [(0usize, 1usize), (1, 0)] {
        assert_eq!(
            a.diagnostics.pinned[g], b.diagnostics.pinned[h],
            "pinned[{g}] vs pinned[{h}]"
        );
    }
    // Conditional modes are per-grouping blocks of unequal width — the case the
    // per-entry swap above cannot cover.
    assert_eq!(a.ranef.len(), BIG + SMALL, "one mode per level");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.ranef[..BIG]), bits(&b.ranef[SMALL..]), "big modes");
    assert_eq!(
        bits(&a.ranef[BIG..]),
        bits(&b.ranef[..SMALL]),
        "small modes"
    );
    for i in 0..n {
        assert_eq!(a.fitted[i].to_bits(), b.fitted[i].to_bits(), "fitted[{i}]");
    }
}

/// Parses `validation/data/empirical/sleepstudy.csv` into the q=2 random-slope
/// design `Reaction ~ 1 + Days + (1 + Days | Subject)`, shared by the
/// warm-start tests below and by the lme4 sleepstudy pins above (warm start,
/// varcorr, offset, weights).
fn sleepstudy_slope_design() -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, GroupIds) {
    let csv = include_str!("../../validation/data/empirical/sleepstudy.csv");
    let mut y = Vec::<f64>::new();
    let mut days = Vec::<f64>::new();
    let mut subj_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // Reaction
        days.push(f[1].parse().unwrap()); // Days
        subj_raw.push(f[2].to_string()); // Subject
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = days[i];
    }
    let (subject, _n_subj) = dense_str(&subj_raw);
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![1],                                 // random slope on Days (col 1)
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: subject,
        extra: vec![],
    };
    (x, y, n, p, model, ids)
}

/// Warm-start forward map, on the sleepstudy slope design: warm-starting
/// at a fit's own reported θ̂ (via [`FitView::theta`], which hands back θ in the
/// CALLER's units) must be a FIXED POINT of the forward map
/// (`LmmGroupings::theta_caller_to_design`). `Days`' scale is clearly off 1.0
/// (values run 0..9), so a dropped forward map moves this test's θ̂ rather than
/// leaving it untouched by construction.
///
/// Reached through the loop tier (`build_workspace`/`fit_on`) rather than
/// `fit_warm`, per the loop-tier warm-start contract this crate carries: the
/// same workspace serves both the cold and the warm call.
#[test]
fn lmm_warm_start_theta_is_a_fixed_point_of_the_forward_map() {
    let (x, y, n, p, model, ids) = sleepstudy_slope_design();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let (sized, sized_ids, perm) = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, perm, n, p, &opts);

    let cold_view = fit_on(&mut ws, &x, &y, &sized_ids, None, &opts);
    let cold_theta = cold_view.theta().to_vec();
    let cold_n_eval = cold_view.n_eval();
    let cold_fit = cold_view.into_fit(&x, &y, &sized_ids, n, p, &model, &opts);
    assert!(
        cold_fit.converged(),
        "cold sleepstudy slope LMM must converge"
    );

    let start = StartValues {
        beta: cold_fit.beta.clone(),
        theta: cold_theta.clone(),
        dispersion: None,
    };
    let warm_view = fit_on(&mut ws, &x, &y, &sized_ids, Some(&start), &opts);
    let warm_theta = warm_view.theta().to_vec();
    let warm_n_eval = warm_view.n_eval();
    let warm_fit = warm_view.into_fit(&x, &y, &sized_ids, n, p, &model, &opts);
    assert!(warm_fit.converged(), "warm-started fit must converge");

    // The contract: a warm start planted at the cold fit's θ̂ (through the
    // forward map) must not end worse than the cold fit; it may end better.
    // BOBYQA stops once its trust radius has shrunk to `RHO_END`, the smallest
    // step it tries, which is not a bound on the distance from its stopping
    // point to the optimum: on a flat valley the incumbent is known only to
    // about √(Δdeviance / curvature), which can be several `RHO_END`, and a
    // restart from it may walk on along the valley.
    assert!(
        warm_fit.deviance <= cold_fit.deviance + 1e-10,
        "the warm restart ended worse than the cold fit: {} vs {}",
        warm_fit.deviance,
        cold_fit.deviance
    );
    // θ̂ stays put, measured against the failure this test exists for: a
    // dropped forward map moves θ̂ by O(1) (`Days` has a column scale of about
    // 5.3), so a band of 1e-4 keeps four orders of magnitude below it.
    // Provenance: with the per-search start radii (cold search at 0.5, warm at
    // 0.1) the restart moved θ[0] by 1.3e-6 with the deviance 5.8e-11 lower
    // (x86_64, 2026-10-02); at 0.1 for both, the worst move was 3.7e-7 on
    // θ[0] with the deviance 2.2e-11 lower.
    const THETA_BAND: f64 = 1e-4;
    for (k, (w, c)) in warm_theta.iter().zip(&cold_theta).enumerate() {
        assert!(
            (w - c).abs() <= THETA_BAND,
            "theta fixed point[{k}]: warm {w} vs cold {c}"
        );
    }
    // Teeth: a warm start planted exactly at θ̂ needs strictly fewer BOBYQA
    // evals than the blind cold start (THETA0 diagonals, 0 off-diagonals) to
    // reach the same point — this is what a dropped forward map (which would
    // instead plant the RESCALED θ̂ read as if it were already in the design's
    // own units) would blow past, since it starts the search somewhere BOBYQA
    // still has to search its way out of.
    assert!(
        warm_n_eval < cold_n_eval,
        "warm start at the true optimum must need fewer evals than the blind \
         cold start: warm {warm_n_eval} vs cold {cold_n_eval}"
    );
}

/// The LMM search is single-stage, so everything lands in stage 2 and the
/// stage-2 count must equal `n_eval`. The shrink count is counter 2 in
/// `crate::counters`' module header on this route.
#[cfg(feature = "counters")]
#[test]
fn lmm_counters_record_stage_two_and_shrink_evals() {
    use crate::counters::Stage;
    let (x, y, n, p, model, ids) = sleepstudy_slope_design();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "sleepstudy must converge");
    let c = f.counters;
    assert_eq!(c.stage_evals[0], 0, "the LMM path runs one stage");
    assert_eq!(
        c.stage_evals[1] as usize, f.n_eval,
        "stage 2 accounts for every eval"
    );
    assert!(
        c.evals_after_last_improve(Stage::Two) < c.stage_evals[1],
        "shrink evals are a strict subset"
    );
    assert_eq!(
        c.pirls_hist.iter().sum::<u32>(),
        0,
        "no PIRLS on a Gaussian LMM"
    );
    assert_eq!(c.agq_evals, 0, "no AGQ on a Gaussian LMM");
}

/// `p == 0` (no fixed-effect columns at all, only a random intercept) is the
/// one shape where `accumulate_lmm_rows` skips `add_rows_multi` entirely, so
/// `n_rows` stays 0 while the row count `n` handed to the weighted deviance
/// correction is not. `reml_deviance` also refuses `p == 0` unconditionally
/// (`suff.n_rows <= p || p == 0`), so this is the crate's other documented
/// degenerate shape (alongside `n <= p`) and the fit reports the same
/// NaN-filled, non-converged contract — this only pins that the weighted
/// correction reaches that contract without panicking on the empty design.
#[test]
fn fit_lmm_p_zero_weighted_reaches_nan_fill() {
    let n_clusters = 4usize;
    let n = 20usize;
    let p = 0usize;
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let ids = GroupIds {
        primary: (0..n).map(|i| (i % n_clusters) as u32).collect(),
        extra: vec![],
    };
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    // Varying, not constant, weights: a wrong `n_rows`-keyed correction and a
    // correct `n`-keyed one both fold in `-Σ log wᵢ`, so this only shows up as
    // "does this branch run to completion", not as a numeric divergence.
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
    let opts = FitOptions {
        target_indices: vec![],
        weights: Some(w),
        ..FitOptions::default()
    };

    let f = fit_cold(&x, &y, n, p, &model, &ids, &opts);

    assert!(!f.converged(), "p=0 is refused, not fit");
    assert_eq!(f.beta.len(), 0);
    assert_eq!(f.se.len(), 0);
    assert!(f.deviance.is_nan(), "deviance {}", f.deviance);
    assert!(f.loglik.is_nan(), "loglik {}", f.loglik);
    assert_eq!(
        f.diagnostics.boundary,
        Boundary::NoOptimum,
        "the p==0 refusal reports NoOptimum, not a silent zero-width fit"
    );
}

/// Fit-level twin of `lmm::tests::maxfun_cap_reports_honest_endpoint`, through
/// `lmm_run_on` + `lmm_view_to_fit`: a `MaxFunReached` cap-out must report
/// finite `tau2`/`varcorr`/`dispersion` (the plateau policy) WITHOUT setting
/// `diagnostics.singular`. It uses TWO variance components (intercept + a
/// slope on column 1), because a single one can never trip the negligible-ratio
/// check. It is built so that the negligible-ratio check
/// (`Fit::has_negligible_component`) — not the boundary pin — is what would
/// flag it: the slope's TRUE random effect is nonzero but tiny relative to
/// the intercept's, so the genuine REML optimum is an INTERIOR point (no θ
/// diagonal ever reaches `PIN_THETA`; `boundary_hit` stays 0 even at full
/// convergence) whose reported standard-deviation ratio is still under
/// `SINGULAR_REL_TOL`. The unlimited-budget optimum at this data (not
/// asserted here, found while writing this test): `varcorr = [0.01278,
/// 6.40e-6, 3.20e-9]`, `singular = true` — confirming the ratio-based check,
/// not a pin, is what marks it singular. A budget short of that endpoint
/// lands on the SAME point (BOBYQA needs one more confirming pass to call it
/// `Converged`), so a single variance component can never exercise this half
/// of the gate — with the gate in place `singular` stays `false` on this
/// `MaxFunReached` point; deleting the `lmm_fit.converged &&` clause below
/// (`fit.diagnostics.singular = fit.diagnostics.singular ||
/// (lmm_fit.converged && fit.has_negligible_component(...))`) flips it to
/// `true` and fails this test — checked by hand while writing it, then
/// restored; not re-checked by CI.
#[test]
fn lmm_maxfun_cap_q2_reports_honest_endpoint_and_not_singular() {
    use bobyqa::{Bobyqa, Config};

    let n_clusters = 40usize;
    let per = 10usize;
    let n = n_clusters * per;
    let p = 2usize;
    let mut st = 13u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids_v = vec![0u32; n];
    for i in 0..n {
        ids_v[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        // The intercept's true random effect is sizeable; the slope's is
        // nonzero but tiny, so the genuine REML optimum sits at a small
        // but INTERIOR variance — no boundary pin ever fires, and any
        // "singular" report has to come from the negligible-ratio check.
        let c = (ids_v[i] as f64) - (n_clusters as f64) / 2.0;
        let re0 = 0.4 * c / (n_clusters as f64);
        let re1 = 0.0002 * c / (n_clusters as f64);
        y[i] = 0.5 + 0.4 * x1 + re0 + re1 * x1 + 0.0002 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    // BOBYQA's cap is the behaviour under test, so the fixture stays off the
    // gradient route: the last cluster keeps a single row (its two-column Z
    // has a singular Gram), which refuses the per-group regression route that
    // BFGS needs (`precompute_group_regressions`).
    let last = (n_clusters - 1) as u32;
    let mut kept = false;
    for id in ids_v.iter_mut() {
        if *id == last {
            if kept {
                *id = last - 1;
            }
            kept = true;
        }
    }
    let ids = GroupIds {
        primary: ids_v,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };

    let (sized, ids, _perm) = spec_sized_from_ids(&model, &ids);
    let mut ws = LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &[1], &[], false);
    let mut x_mat = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }
    ws.suff_mut().reset();
    ws.suff_mut()
        .add_rows_multi(x_mat.as_ref(), &y, &ids.primary, &[], None);

    let n_theta = ws.theta.len();
    let npt = 2 * n_theta + 1; // PRIMA's minimum npt (n_theta == 3 here)
    let config = {
        let mut c = Config::new(n_theta);
        c.npt = npt;
        // 41 evals short of the unlimited-budget optimum (191 evals on this
        // fixture, probed 2026-10-07 once the single-row cluster above kept it
        // on BOBYQA): the reported point already carries a negligible
        // slope-to-intercept stddev ratio (2.5e-4) well before BOBYQA's own
        // Converged confirmation fires.
        c.max_fun = 150;
        c
    };
    ws.solver = Bobyqa::new(n_theta, config).expect("legal minimal config");

    let fit = {
        let v = lmm_run_on(
            &mut ws,
            &opts.target_indices,
            None,
            None,
            ColumnScaling::IDENTITY,
        );
        lmm_view_to_fit(&v, &x, &ids, n, p, &opts)
    };

    assert!(!fit.converged(), "a capped fit must not report converged");
    assert!(
        !fit.tau2.is_empty() && fit.tau2.iter().all(|v| v.is_finite()),
        "plateau policy: capped endpoint must report tau2, got {:?}",
        fit.tau2
    );
    assert!(
        !fit.varcorr.is_empty() && fit.varcorr.iter().flatten().all(|v| v.is_finite()),
        "plateau policy: capped endpoint must report varcorr, got {:?}",
        fit.varcorr
    );
    assert!(
        fit.dispersion.is_finite(),
        "plateau policy: capped endpoint must report a finite dispersion"
    );
    // The point IS negligible by the ratio check's own rule — this is what
    // makes the gate load-bearing here, unlike a single-component fixture
    // whose one variance is never small enough to trip it either way.
    let sd: Vec<f64> = fit.varcorr[0]
        .iter()
        .enumerate()
        .filter(|(i, _)| [0, 2].contains(i)) // q=2 vech diagonals: [Σ00, Σ10, Σ11]
        .map(|(_, &v)| v.sqrt())
        .collect();
    let ratio = sd[1] / sd[0];
    assert!(
        ratio <= 1e-3,
        "fixture must be genuinely negligible by SINGULAR_REL_TOL, ratio {ratio}"
    );
    assert!(
        !fit.singular(),
        "a capped endpoint reports singular = false, even though its varcorr \
         ratio ({ratio}) is negligible — has_negligible_component must stay \
         gated on `converged`"
    );
}

/// What `dense_accumulation_case` returns: a dense workspace, the
/// column-major design, `y`, the primary ids and the prior weights.
type DenseCase = (LmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, Option<Vec<f64>>);

/// A random-intercept design for the dense accumulation tests: `n` rows in 10
/// clusters, columns `[1, x1, x2]`; `x2` is independent of `x1`, or with
/// `collinear` equal to it up to noise of size 1e-4 (pivot ratio about 1e-7,
/// far under `GRAM_PIVOT_MIN`). With `weighted`, prior weights in
/// [0.75, 1.25], whose `weight_scale` is exactly 1, so the kernel sees them as
/// they are.
fn dense_accumulation_case(n: usize, collinear: bool, weighted: bool) -> DenseCase {
    let (p, k) = (3usize, 10usize);
    let mut st = 46u64;
    let x = Mat::<f64>::from_fn(n, p, |_, j| if j == 0 { 1.0 } else { lcg(&mut st) });
    let x = if collinear {
        let noise: Vec<f64> = (0..n).map(|_| 1e-4 * lcg(&mut st)).collect();
        Mat::<f64>::from_fn(n, p, |i, j| {
            if j == 2 {
                x[(i, 1)] + noise[i]
            } else {
                x[(i, j)]
            }
        })
    } else {
        x
    };
    let ids: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.5 + x[(i, 1)] - 0.3 * x[(i, 2)] + 0.2 * ids[i] as f64 + 0.1 * lcg(&mut st))
        .collect();
    let w = weighted.then(|| {
        (0..n)
            .map(|_| 1.0 + 0.25 * lcg(&mut st))
            .collect::<Vec<f64>>()
    });
    if let Some(w) = &w {
        assert_eq!(crate::family::weight_scale(Some(w), n), 1.0);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: k as u32,
            },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let gids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let (sized, gids, _perm) = spec_sized_from_ids(&model, &gids);
    let ws = LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &[], &[], false);
    (ws, x, y, gids.primary.clone(), w)
}

/// The dense route's two ways to the fixed-effect factor `U`
/// (`lmm::DesignQr`). On a well-conditioned design it is read off the
/// accumulated Gram: `X̃'X̃` is then exactly diagonal and every statistic
/// matches the Givens route's to round-off. On a design under
/// `GRAM_PIVOT_MIN` the route is the Givens pass itself, bit for bit.
#[test]
fn accumulate_lmm_rows_gram_route_matches_the_givens_route() {
    let (n, p) = (300usize, 3usize);
    for (collinear, weighted) in [(false, false), (false, true), (true, false), (true, true)] {
        let tag = format!("collinear={collinear} weighted={weighted}");
        let (mut ws, x, y, ids, w) = dense_accumulation_case(n, collinear, weighted);
        let (mut givens, ..) = dense_accumulation_case(n, collinear, weighted);
        accumulate_lmm_rows(
            &mut ws,
            x.as_ref(),
            &y,
            n,
            p,
            &ids,
            &[],
            w.as_deref(),
            false,
            crate::lmm::Criterion::Reml,
        );
        let b = givens.suff_mut();
        b.reset();
        b.set_design_qr(x.as_ref(), w.as_deref());
        b.add_rows_multi(x.as_ref(), &y, &ids, &[], w.as_deref());
        let a = ws.suff_mut();
        let m = p + 1;
        if collinear {
            let bits = |s: &[f64]| s.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(a.design_qr.r_diag()),
                bits(b.design_qr.r_diag()),
                "{tag}: D"
            );
            for j in 0..m {
                assert_eq!(
                    bits(a.c.col_as_slice(j)),
                    bits(b.c.col_as_slice(j)),
                    "{tag}: c col {j}"
                );
            }
            for col in 0..a.s.ncols() {
                assert_eq!(
                    bits(a.s.col_as_slice(col)),
                    bits(b.s.col_as_slice(col)),
                    "{tag}: s col {col}"
                );
            }
            continue;
        }
        for j in 0..p {
            for i in (j + 1)..p {
                assert_eq!(a.c[(i, j)], 0.0, "{tag}: X̃'X̃ off the diagonal");
            }
        }
        for (j, (&da, &db)) in a
            .design_qr
            .r_diag()
            .iter()
            .zip(b.design_qr.r_diag())
            .enumerate()
        {
            assert!(
                (da - db).abs() <= 1e-12 * db.abs(),
                "{tag}: D[{j}] {da} vs {db}"
            );
        }
        for j in 0..m {
            for i in j..m {
                let scale = (b.c[(i, i)] * b.c[(j, j)]).sqrt();
                let d = (a.c[(i, j)] - b.c[(i, j)]).abs();
                assert!(
                    d <= 1e-10 * scale,
                    "{tag}: c[{i},{j}] {} vs {}",
                    a.c[(i, j)],
                    b.c[(i, j)]
                );
            }
        }
        for col in 0..a.s.ncols() {
            for j in 0..m {
                let scale = (b.c[(j, j)] * b.counts[col]).sqrt();
                let d = (a.s[(j, col)] - b.s[(j, col)]).abs();
                assert!(
                    d <= 1e-10 * scale,
                    "{tag}: s[{j},{col}] {} vs {}",
                    a.s[(j, col)],
                    b.s[(j, col)]
                );
            }
        }
    }
}

/// A weighted dense LMM draw on a reused workspace allocates nothing in the
/// accumulation: the scaled weights ŵ = w/s live in the workspace, not in a
/// per-call vector.
///
/// Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests
/// accumulate_lmm_rows_weighted_reuse_allocates_nothing -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn accumulate_lmm_rows_weighted_reuse_allocates_nothing() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: u64 = 10;
    let (n, p) = (300usize, 3usize);
    let (mut ws, x, y, ids, w) = dense_accumulation_case(n, false, true);
    let w = w.unwrap();
    let run = |ws: &mut LmmWorkspace| {
        accumulate_lmm_rows(
            ws,
            x.as_ref(),
            &y,
            n,
            p,
            &ids,
            &[],
            Some(&w),
            false,
            crate::lmm::Criterion::Reml,
        );
    };
    run(&mut ws);
    crate::test_support::settle_background_allocs();
    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        run(&mut ws);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert_eq!(
        stats.total_blocks, 0,
        "{N_CALLS} weighted accumulations allocated {} blocks",
        stats.total_blocks
    );
}

/// What `sparse_accumulation_draw` returns: the column-major design, `y`, the
/// primary ids, the extra ids and the prior weights.
type SparseDraw = (Mat<f64>, Vec<f64>, Vec<u32>, Vec<Vec<u32>>, Vec<f64>);

/// One draw of a primary-plus-crossed random-intercept design: `n` rows,
/// primary level `i % 8`, crossed level `(i / shift) % 5`, columns
/// `[1, x1, x2]` and prior weights in [0.75, 1.25], all drawn from `seed`.
/// Draws with the same `shift` share their rows' levels.
fn sparse_accumulation_draw(n: usize, shift: usize, seed: u64) -> SparseDraw {
    let mut st = seed;
    let x = Mat::<f64>::from_fn(n, 3, |_, j| if j == 0 { 1.0 } else { lcg(&mut st) });
    let ids: Vec<u32> = (0..n).map(|i| (i % 8) as u32).collect();
    let extra = vec![(0..n)
        .map(|i| ((i / shift) % 5) as u32)
        .collect::<Vec<u32>>()];
    let y: Vec<f64> = (0..n)
        .map(|i| {
            0.5 + x[(i, 1)] - 0.3 * x[(i, 2)] + 0.2 * ids[i] as f64 - 0.1 * extra[0][i] as f64
                + 0.1 * lcg(&mut st)
        })
        .collect();
    let w: Vec<f64> = (0..n).map(|_| 1.0 + 0.25 * lcg(&mut st)).collect();
    (x, y, ids, extra, w)
}

/// A sparse-kernel workspace for [`sparse_accumulation_draw`]'s design.
fn sparse_accumulation_workspace(n: usize, ids: &[u32], extra: &[Vec<u32>]) -> LmmWorkspace {
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 5 },
                slopes: vec![],
            }],
        }),
    };
    let gids = GroupIds {
        primary: ids.to_vec(),
        extra: extra.to_vec(),
    };
    let (sized, gids, perm) = spec_sized_from_ids(&model, &gids);
    assert!(perm.is_identity() && gids.primary == ids && gids.extra == extra);
    LmmWorkspace::for_cluster_spec_ext(3, &sized, n, &[], &[], true)
}

/// A sparse-kernel workspace that is refilled in place for a new draw on the
/// rows its pattern was built from, and rebuilt for a draw on other rows,
/// holds what a workspace built fresh for that draw holds: the deviance at a
/// fixed θ, which reads every accumulated Gram, agrees bit for bit. The draws
/// change x, y and the weights, and one is unweighted.
#[test]
fn accumulate_lmm_rows_sparse_reuse_matches_a_fresh_workspace() {
    let (n, p) = (240usize, 3usize);
    let theta = [0.7, 0.4];
    let mut reused: Option<LmmWorkspace> = None;
    for (shift, seed, weighted) in [(3, 1u64, true), (3, 2, true), (3, 3, false), (7, 4, true)] {
        let (x, y, ids, extra, w) = sparse_accumulation_draw(n, shift, seed);
        let w = weighted.then_some(&w[..]);
        let ws = reused.get_or_insert_with(|| sparse_accumulation_workspace(n, &ids, &extra));
        accumulate_lmm_rows(
            ws,
            x.as_ref(),
            &y,
            n,
            p,
            &ids,
            &extra,
            w,
            false,
            crate::lmm::Criterion::Reml,
        );
        let mut fresh = sparse_accumulation_workspace(n, &ids, &extra);
        accumulate_lmm_rows(
            &mut fresh,
            x.as_ref(),
            &y,
            n,
            p,
            &ids,
            &extra,
            w,
            false,
            crate::lmm::Criterion::Reml,
        );
        let (a, b) = (ws.kernel.deviance(&theta), fresh.kernel.deviance(&theta));
        assert!(a.is_finite(), "shift={shift} seed={seed}: deviance {a}");
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "shift={shift} seed={seed}: {a} vs {b}"
        );
    }
}

/// A weighted draw on a reused sparse-kernel workspace, on the rows its
/// pattern was built from, allocates nothing in the accumulation: the Grams
/// are refilled in the workspace's own buffers, not rebuilt.
///
/// Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests
/// accumulate_lmm_rows_sparse_weighted_reuse_allocates_nothing -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn accumulate_lmm_rows_sparse_weighted_reuse_allocates_nothing() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: u64 = 10;
    let (n, p) = (240usize, 3usize);
    let (x, y, ids, extra, w) = sparse_accumulation_draw(n, 3, 1);
    let mut ws = sparse_accumulation_workspace(n, &ids, &extra);
    let run = |ws: &mut LmmWorkspace| {
        accumulate_lmm_rows(
            ws,
            x.as_ref(),
            &y,
            n,
            p,
            &ids,
            &extra,
            Some(&w),
            false,
            crate::lmm::Criterion::Reml,
        );
    };
    run(&mut ws);
    crate::test_support::settle_background_allocs();
    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        run(&mut ws);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert_eq!(
        stats.total_blocks, 0,
        "{N_CALLS} weighted sparse accumulations allocated {} blocks",
        stats.total_blocks
    );
}

// --- curvature hints ---

/// `y` with a deterministic per-row shift, a second draw of the same design.
fn shifted(y: &[f64], k: f64) -> Vec<f64> {
    y.iter()
        .enumerate()
        .map(|(i, &v)| v + k * ((i % 7) as f64 - 3.0))
        .collect()
}

fn fit_bits(v: &super::core::FitView<'_>) -> (u64, Vec<u64>, usize) {
    (
        v.betas()[0].to_bits() ^ v.betas()[1].to_bits(),
        v.theta().iter().map(|t| t.to_bits()).collect(),
        v.n_eval(),
    )
}

/// A hinted fit is a function of (data, start, hint) only: the same draws in
/// either order on one workspace, and each on a fresh workspace, agree bit
/// for bit, and the hint is still the pilot's after the loop.
#[test]
fn curvature_hint_results_do_not_depend_on_call_order() {
    use super::core::{curvature_hint, fit_on_hinted};
    let (x, y, n, p, model, ids) = sleepstudy_slope_design();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let (sized, sized_ids, perm) = spec_sized_from_ids_pub(&model, &ids);
    let build = || build_workspace(&sized, perm, n, p, &opts);
    let mut pilot_ws = build();
    let pilot = fit_on(&mut pilot_ws, &x, &y, &sized_ids, None, &opts);
    assert!(pilot.diagnostics().converged);
    let hint = curvature_hint(&mut pilot_ws).expect("sleepstudy takes the gradient route");
    assert_eq!(hint.dim(), 3);
    let snapshot = format!("{hint:?}");
    let (y1, y2) = (shifted(&y, 0.3), shifted(&y, -0.2));

    let mut ab = build();
    let a1 = fit_bits(&fit_on_hinted(
        &mut ab,
        &x,
        &y1,
        &sized_ids,
        None,
        Some(&hint),
        &opts,
    ));
    let a2 = fit_bits(&fit_on_hinted(
        &mut ab,
        &x,
        &y2,
        &sized_ids,
        None,
        Some(&hint),
        &opts,
    ));
    let mut ba = build();
    let b2 = fit_bits(&fit_on_hinted(
        &mut ba,
        &x,
        &y2,
        &sized_ids,
        None,
        Some(&hint),
        &opts,
    ));
    let b1 = fit_bits(&fit_on_hinted(
        &mut ba,
        &x,
        &y1,
        &sized_ids,
        None,
        Some(&hint),
        &opts,
    ));
    let f1 = fit_bits(&fit_on_hinted(
        &mut build(),
        &x,
        &y1,
        &sized_ids,
        None,
        Some(&hint),
        &opts,
    ));
    assert_eq!(a1, b1);
    assert_eq!(a2, b2);
    assert_eq!(a1, f1);
    assert_eq!(
        format!("{hint:?}"),
        snapshot,
        "the fit never writes the hint"
    );
}

/// The hint only changes the search path: a hinted fit reaches the unhinted
/// fit's optimum.
#[test]
fn curvature_hint_reaches_the_unhinted_optimum() {
    use super::core::{curvature_hint, fit_on_hinted};
    let (x, y, n, p, model, ids) = sleepstudy_slope_design();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let (sized, sized_ids, perm) = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, perm, n, p, &opts);
    let _ = fit_on(&mut ws, &x, &y, &sized_ids, None, &opts);
    let hint = curvature_hint(&mut ws).unwrap();
    let y1 = shifted(&y, 0.3);
    let plain = fit_on(&mut ws, &x, &y1, &sized_ids, None, &opts)
        .into_fit(&x, &y1, &sized_ids, n, p, &model, &opts)
        .deviance;
    let v = fit_on_hinted(&mut ws, &x, &y1, &sized_ids, None, Some(&hint), &opts);
    assert!(v.diagnostics().converged);
    let hinted = v
        .into_fit(&x, &y1, &sized_ids, n, p, &model, &opts)
        .deviance;
    assert!(
        hinted <= plain + 1e-9,
        "hinted {hinted} vs unhinted {plain}"
    );
}

/// A hint taken at another θ dimension is rejected.
#[test]
#[should_panic(expected = "curvature hint dimension")]
fn curvature_hint_of_another_dimension_is_rejected() {
    use super::core::{curvature_hint, fit_on_hinted};
    let (x, y, n, p, model, ids) = sleepstudy_slope_design();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let (sized, sized_ids, perm) = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, perm, n, p, &opts);
    let _ = fit_on(&mut ws, &x, &y, &sized_ids, None, &opts);
    let hint = curvature_hint(&mut ws).unwrap();
    // The same data as a random-intercept model: n_θ = 1.
    let intercept = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let (sized1, ids1, perm1) = spec_sized_from_ids_pub(&intercept, &ids);
    let mut ws1 = build_workspace(&sized1, perm1, n, p, &opts);
    let _ = fit_on_hinted(&mut ws1, &x, &y, &ids1, None, Some(&hint), &opts);
}

/// No hint where the fit has no gradient: a random-intercept LMM searches
/// with BOBYQA.
#[test]
fn curvature_hint_is_none_off_the_gradient_route() {
    use super::core::curvature_hint;
    let (x, y, n, p, _, ids) = sleepstudy_slope_design();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let intercept = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let (sized, sized_ids, perm) = spec_sized_from_ids_pub(&intercept, &ids);
    let mut ws = build_workspace(&sized, perm, n, p, &opts);
    let _ = fit_on(&mut ws, &x, &y, &sized_ids, None, &opts);
    assert!(curvature_hint(&mut ws).is_none());
}

/// `y ~ 1 + x + (1 | g)`, 10 groups of 6: the fixture of the Satterthwaite
/// note tests. Returns `(x, y, ids, n, p)`.
fn satterthwaite_note_data() -> (Vec<f64>, Vec<f64>, GroupIds, usize, usize) {
    let mut st = 21u64;
    let (mut x, mut y, mut g) = (vec![], vec![], vec![]);
    for grp in 0..10u32 {
        let b = lcg(&mut st);
        for _ in 0..6 {
            let x1 = lcg(&mut st);
            x.extend([1.0, x1]);
            y.push(0.3 + 0.5 * x1 + b + 0.5 * lcg(&mut st));
            g.push(grp);
        }
    }
    let n = y.len();
    let ids = GroupIds {
        primary: g,
        extra: vec![],
    };
    (x, y, ids, n, 2)
}

/// A `MaxFunReached` endpoint gets its df, and the note says the endpoint
/// did not converge.
#[test]
fn note_on_maxfun_endpoint() {
    use bobyqa::{Bobyqa, Config};

    let (x, y, ids, n, p) = satterthwaite_note_data();
    let model = intercept_only_spec(Sizing::FixedClusters { n_clusters: 1 });
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let (sized, ids, perm) = spec_sized_from_ids(&model, &ids);
    let mut ws = build_workspace(&sized, perm, n, p, &opts);
    {
        let (lmm, _) = ws.lmm_parts().expect("a Gaussian mixed workspace");
        // One θ, so the search is BOBYQA's; five evaluations stop it short.
        let mut config = Config::new(1);
        config.max_fun = 5;
        lmm.solver = Bobyqa::new(1, config).expect("legal minimal config");
    }
    let mut fit =
        fit_on(&mut ws, &x, &y, &ids, None, &opts).into_fit(&x, &y, &ids, n, p, &model, &opts);
    assert_eq!(fit.diagnostics.boundary, Boundary::NoOptimum);
    assert!(fit.deviance.is_finite());
    super::core::attach_satterthwaite(&mut ws, &mut fit, &opts);
    assert!(
        fit.coef_df.iter().all(|d| d.is_finite()),
        "{:?}",
        fit.coef_df
    );
    let note = fit
        .diagnostics
        .notes
        .iter()
        .find(|n| matches!(n, Note::SatterthwaiteCurvature { .. }));
    assert!(
        matches!(
            note,
            Some(Note::SatterthwaiteCurvature {
                converged: false,
                nonfinite: false,
                ..
            })
        ),
        "{note:?}"
    );
}

/// A direction of `H_p` the pseudo-inverse drops raises the note; a positive
/// definite `H_p` at a converged endpoint raises none. `H_p` is read off an
/// exact quadratic at steps of 1/2, so every difference is exact.
#[test]
fn note_on_dropped_direction() {
    use crate::lmm::satterthwaite::{assemble, Collected};

    let state = |h: [[f64; 2]; 2]| {
        let f = |a: f64, b: f64| 0.5 * (h[0][0] * a * a + 2.0 * h[0][1] * a * b + h[1][1] * b * b);
        let s = 0.5;
        assemble(&Collected {
            free: vec![0, 1],
            h: vec![s, s],
            f0: 0.0,
            f_axis: vec![[f(s, 0.0), f(-s, 0.0)], [f(0.0, s), f(0.0, -s)]],
            f_mixed: vec![[f(s, s), f(-s, -s)]],
            g_axis: None,
            c0: Mat::identity(1, 1),
            c_axis: vec![
                [Mat::identity(1, 1), Mat::identity(1, 1)],
                [Mat::identity(1, 1), Mat::identity(1, 1)],
            ],
            tau: 1.0,
            nu: 10.0,
        })
    };
    let singular = state([[1.0, 1.0], [1.0, 1.0]]);
    assert!(
        matches!(
            super::core::curvature_note(&singular, true),
            Some(Note::SatterthwaiteCurvature {
                dropped: 1,
                negative: 0,
                converged: true,
                nonfinite: false,
            })
        ),
        "dropped {} negative {}",
        singular.dropped,
        singular.negative
    );
    let clean = state([[2.0, 1.0], [1.0, 2.0]]);
    assert!(super::core::curvature_note(&clean, true).is_none());
}

/// A non-finite deviance on the stencil (here forced through θ̂ itself) fails
/// the pass for the whole fit: every `coef_df` NaN, the note with `nonfinite`,
/// and the contrasts' `NoDf(Failed)`.
#[test]
fn failed_pass_gives_no_df_failed() {
    use crate::inference::contrast::{ContrastError, NoDfReason};

    let (x, y, ids, n, p) = satterthwaite_note_data();
    let model = intercept_only_spec(Sizing::FixedClusters { n_clusters: 1 });
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let (mut fit, mut ws) = fit_cold_keep_workspace(&x, &y, n, p, &model, &ids, &opts);
    assert!(fit.converged());
    ws.lmm_parts().expect("a Gaussian mixed workspace").0.theta[0] = f64::NAN;
    super::core::attach_satterthwaite(&mut ws, &mut fit, &opts);
    assert!(fit.coef_df.iter().all(|d| d.is_nan()));
    assert!(fit.diagnostics.notes.iter().any(|n| matches!(
        n,
        Note::SatterthwaiteCurvature {
            nonfinite: true,
            ..
        }
    )));
    assert_eq!(
        fit.contrast_t(&[0.0, 1.0]),
        Err(ContrastError::NoDf(NoDfReason::Failed))
    );
    assert_eq!(
        fit.contrast_f(&[1.0, 0.0, 0.0, 1.0], 2),
        Err(ContrastError::NoDf(NoDfReason::Failed))
    );
}

/// What the ports read after a failed pass (a forced non-finite stencil
/// point, as `failed_pass_gives_no_df_failed`): `df_method = "failed"`, no t
/// p-value, and normal intervals on every coefficient although the SEs are
/// finite.
#[cfg(feature = "orchestrate")]
#[test]
fn failed_pass_gives_failed_port_fields_and_normal_intervals() {
    use crate::inference::tails::normal_quantile;

    let (x, y, ids, n, p) = satterthwaite_note_data();
    let model = intercept_only_spec(Sizing::FixedClusters { n_clusters: 1 });
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let (mut fit, mut ws) = fit_cold_keep_workspace(&x, &y, n, p, &model, &ids, &opts);
    ws.lmm_parts().expect("a Gaussian mixed workspace").0.theta[0] = f64::NAN;
    super::core::attach_satterthwaite(&mut ws, &mut fit, &opts);

    let (coef_p, df_method) = crate::orchestrate::t_tests(&fit, &opts.target_indices);
    assert_eq!(df_method, "failed");
    assert!(coef_p.iter().all(|v| v.is_nan()));
    assert!(fit.se.iter().all(|s| s.is_finite()));
    let ci = crate::orchestrate::confint(&fit.beta, &fit.se, &fit.coef_df, 0.95).unwrap();
    for (i, [lo, hi]) in ci.into_iter().enumerate() {
        assert_eq!(lo, fit.beta[i] + normal_quantile(0.025) * fit.se[i]);
        assert_eq!(hi, fit.beta[i] + normal_quantile(0.975) * fit.se[i]);
    }
}
