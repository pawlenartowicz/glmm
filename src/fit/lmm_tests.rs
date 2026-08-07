//! LMM estimator tests (`Family::Gaussian`, `re: Some`), plus the
//! `loop_advanced`-gated LMM sweep/refit dev-seam tests.

use super::*;
// The loop-tier entries reached through the module rather than the re-export,
// which is `loop_advanced`-gated: the pivot the region-2 tests below assert on
// is recorded on every route, so these must run under default features too.
use super::core::{build_workspace, fit_on};
use crate::lmm::{fit_lmm, LmmWorkspace};
use crate::{
    Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing, StartValues,
};
use faer::Mat;

#[cfg(feature = "loop_advanced")]
use super::common_tests::lmm_hand_dataset;
use super::common_tests::{assert_pinned, dense_str, lcg, PIN_REL_ITER};
// The dev seam is not on the `loop_advanced` public surface, so this equivalence
// test reaches it directly from its module.
#[cfg(feature = "loop_advanced")]
use super::loop_advanced_seam::{build_lmm_workspace, refit_lmm};

use super::lmm::{lmm_run_on, lmm_view_to_fit};
use crate::test_support::{assert_near, intercept_only_spec};

/// `lmm_run_on` + `lmm_view_to_fit` on a hand-accumulated workspace must
/// reproduce the `Fit` that `fit_cold` produces for the same single-random-
/// intercept Gaussian LMM — pins the view/mapper split as behavior-preserving.
#[test]
fn lmm_run_on_view_maps_to_same_fit_as_fit_cold() {
    let n_clusters = 6usize;
    let per = 8usize;
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
        let re = 0.3 * ((ids_v[i] as f64) - (n_clusters as f64) / 2.0);
        y[i] = 0.5 + 0.4 * x1 + re + 0.2 * lcg(&mut st);
    }
    let model = ModelSpec {
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
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };

    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);

    let sized = spec_sized_from_ids(&model, &ids);
    let mut ws = LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &[], &[]);
    let mut x_mat = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }
    ws.suff.reset();
    ws.suff
        .add_rows_multi(x_mat.as_ref(), &y, &ids.primary, &[], None);
    let via = {
        let v = lmm_run_on(&mut ws, &opts.target_indices, None);
        lmm_view_to_fit(&v, &x, &ids, n, p, &opts)
    };
    assert_near(&cold.beta, &via.beta, "beta");
    assert_near(&cold.tau2, &via.tau2, "tau2");
    assert_near(&[cold.dispersion], &[via.dispersion], "dispersion");
    assert_near(&cold.se, &via.se, "se");
}

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
// imprecision is a large standard error. Both used to be discarded — one
// NaN-filled, the other had a column silently dropped — by a rank guard whose
// statistic (`min|L_ii| / max|L_ii|` on X'V⁻¹X) measured column SCALE rather
// than collinearity. That guard is gone: the dense-LMM route refuses no design
// on conditioning grounds at all, and instead records the scale-invariant
// per-column pivot ratio for the diagnostics channel to flag below
// `lmm::PIVOT_MIN = 1e-12`. Neither design reaches even that.
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
/// ratio here is O(1) (1.0, 0.235, 1.0) — the columns are mutually
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
/// `validation/prep/gen_illcond_data.R`, and `tests/validation_oracle.rs` bands
/// its β, SE, σ̂, log-likelihood and variance components against lme4 on the FULL
/// three-column design at `validation/tol.R`'s cross-engine tolerances. That
/// golden exists because of this change: while the design NaN-filled there was
/// nothing for a reference to agree with. The two engines land within 6e-11 on β
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
    // note on the stable surface, and a recorded pivot ratio nowhere near
    // `PIVOT_MIN`. Without this the 0.235 quoted above could drift to anything
    // and every other assertion here would still pass.
    assert!(
        f.diagnostics.notes.is_empty(),
        "no column is entangled here, so no note may be raised: {:?}",
        f.diagnostics.notes
    );
    let sized = spec_sized_from_ids_pub(&intercept_only_lmm(), &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    let d = fit_on(&mut ws, &x, &y, &ids, None, &opts).diagnostics();
    assert!(!d.ill_conditioned, "pivot {} must clear the floor", d.pivot);
    // Band, not a pin: the doc comment above quotes 0.235 as the minimum
    // per-column pivot ratio, and the point is that it is O(1) rather than the
    // 1.6e-10 the old min/max statistic reported on the same fit.
    assert!(
        (0.2..0.3).contains(&d.pivot),
        "min pivot ratio must stay at the quoted 0.235, got {}",
        d.pivot
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
/// same size. That is the deliverable. It is also where lme4 already was — it
/// fits all four columns of this design with the same ±3.8e7 blow-up — so the
/// crate now agrees with the reference on the column set instead of returning a
/// three-column model lme4 never proposed.
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
/// itself assertable: until this release the crate returned a three-column
/// model lme4 never proposed, so there was nothing to compare the pair against
/// and the test could only band the unentangled columns and the identified sum
/// against lme4 on an explicitly-reduced design. Bands are `validation/tol.R`'s
/// cross-engine ones, unchanged.
///
/// The provenance, since these constants are frozen in-crate rather than under
/// `validation/goldens/`: the design is emitted as
/// `validation/data/simulated/sim_entangled_pair_lmm.csv` by
/// `validation/prep/gen_illcond_data.R`, whose generator arithmetic is
/// bit-identical to [`build_gap_a_salvage_design`] above (verified over all 4000
/// doubles) and whose CSV round-trips exactly at 17 significant digits. The
/// reference is `lmer(y ~ 1 + t + v + z + (1|g), data, REML = TRUE)` under lme4
/// 1.1.38 / R 4.5.3.
///
/// It is NOT registered in `validation/manifest.json` as a cross-engine golden,
/// and the reason is worth stating rather than leaving to be rediscovered. Every
/// quantity below agrees inside its band, but the REML criterion does not:
/// glmm reports −239.09477 against lme4's −239.09437, a gap of 4.0e-4 where
/// `tol.R`'s `loglik_abs_lmm` is an absolute 2e-6. That band was calibrated on
/// well-conditioned designs.
///
/// The cause was measured rather than inferred, by re-evaluating this design's
/// REML criterion in 60-digit arithmetic from its closed form for one balanced
/// intercept RE — V_j = σ²I_m + τ²11', so V_j⁻¹ = (I − c·11')/σ² with
/// c = τ²/(σ² + mτ²) and log|V_j| = (m−1)log σ² + log(σ² + mτ²) — and comparing
/// term by term against the same evaluation carried out at reduced precision:
///
///   * the two θ̂ are not what separates the engines. The exact criterion at
///     glmm's θ̂ and at lme4's differs by 5.1e-11; the objective really is flat
///     here. (Two supporting controls agree: on the reduced design the two
///     engines match the criterion to 1.0e-10, and lme4 returns the identical
///     full-design value under a 1e-14-tightened optimizer, so neither side is
///     merely under-converged.)
///   * `log|X'V⁻¹X|` is where the digits go, and "loses digits" understates it:
///     at float64 working precision this design's 4×4 X'V⁻¹X is numerically
///     singular, and its log-determinant does not stabilise until roughly 25
///     decimal digits. Neither engine can evaluate that term to the 2e-6 an
///     absolute band assumes.
///   * so both engines miss the criterion's exact value, −239.0944407: lme4 by
///     +7.0e-5, glmm by −3.3e-4. glmm is the further of the two, by 4.7×. That
///     is recorded as a finding, not argued away — but it is a shared
///     consequence of the conditioning, not a difference of method.
///
/// Registering the rung therefore needs the band question settled first, which
/// is a calibration decision, not a test edit.
#[test]
fn lmm_entangled_pair_fits_in_full_with_honest_ses() {
    // lme4 1.1.38 on the FULL design [1, t, v, z], REML. Written in the shortest
    // decimal form that round-trips to the same f64 as lme4's 17-digit output —
    // the same doubles, not truncated ones; padding them back out is a clippy
    // `excessive_precision` error and changes nothing. Measured agreement with
    // glmm, worst per row: β 5.7e-4 (both entangled columns), SE 2.8e-4 (same
    // two), stddev 1.4e-6, σ̂ 8.5e-8, β_t + β_v 4.8e-7. The two entangled cells
    // are the tightest in the crate against a 1e-3 band — 1.8× margin — and that
    // is the honest size of the disagreement, not a slack to be traded away:
    // the pair is by construction the least-determined direction in the design,
    // so it is where two independent implementations differ most.
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
    // validation/tol.R: beta_rel, se_rel, stddev_rel.
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
    // Compare on the STDDEV scale, which is what tol.R's stddev_rel bands.
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
    let sized = spec_sized_from_ids_pub(&intercept_only_lmm(), &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
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
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: subject,
        extra: vec![],
    };
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
            },
        ),
        // Well off θ̂ ≈ [0.97, 0.015, 0.23] in every coordinate; the LMM
        // path threads θ only (β is solved exactly given θ).
        (
            "perturbed",
            StartValues {
                beta: vec![0.0; p],
                theta: vec![3.0, 0.5, 1.5],
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
        x[i * p] = 1.0; // intercept
        x[i * p + 1] = days[i]; // Days
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
    assert!(
        (f.se[0] - REF_SE0).abs() / REF_SE0 < 2e-2,
        "se0 {} vs {REF_SE0}",
        f.se[0]
    );
    assert!(
        (f.se[1] - REF_SE1).abs() / REF_SE1 < 2e-2,
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
}

/// Campaign instrumentation: `fit` must surface the optimizer eval count, the
/// minimized criterion, and boundary/singular status. Oracle: lme4's frozen
/// sleepstudy REML fit — REMLcrit = glmm deviance + df·(1 + ln 2π), df = n − p
/// (glmm's reml_deviance omits the df·(1+ln 2π) constant lme4's REMLcrit
/// carries; loglik = −REMLcrit/2 is what results/lme4_empirical stores).
#[test]
fn fit_exposes_n_eval_deviance_singular() {
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
        x[i * p] = 1.0; // intercept
        x[i * p + 1] = days[i]; // Days
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

    assert!(f.n_eval > 0, "BOBYQA ran, evals must be counted");
    assert!(f.deviance.is_finite());
    assert!(!f.singular(), "sleepstudy is an interior optimum");
    let n = 180.0_f64;
    let p = 2.0_f64; // intercept + Days
    let df = n - p;
    let lme4_loglik = -871.814135979976; // validation/results/lme4_empirical/sleepstudy.json .estimates.loglik
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
    let o: Vec<f64> = (0..n).map(|i| 5.0 * (i % 4) as f64).collect();

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![1],
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
            primary: subject,
            extra: vec![],
        },
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
/// unweighted case (see `lme::profiled_deviance`) PLUS the weighted Gaussian log-density's
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
        x[i * p] = 1.0; // intercept
        x[i * p + 1] = days[i]; // Days
    }
    let (subject, _n_subj) = dense_str(&subj_raw);
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();

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
    let sized = spec_sized_from_ids(&model, &ids);
    let mut ws = LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &[1], &[]);
    let mut x_mat = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }
    ws.suff
        .add_rows_multi(x_mat.as_ref(), &y, &ids.primary, &[], Some(&w));
    let lmm_fit = fit_lmm(&mut ws, &[0, 1], None);
    let sigma = lmm_fit.sigma_sq.sqrt();
    assert!(
        (sigma - REF_SIGMA).abs() / REF_SIGMA < 1e-4,
        "sigma {sigma} vs {REF_SIGMA}"
    );
}

/// Task 5: constant weights (w ≡ 2) must reproduce the unweighted fit's β,
/// SE, AND tau2 exactly (1e-10) — under w ≡ c, the substitution θ̃ = √c·θ
/// maps the weighted profiled deviance onto the unweighted one 1:1, so θ̂
/// scales by 1/√c while σ̂² scales by c, and tau2 = θ²σ̂² is invariant.
/// Verified against lme4 separately: sleepstudy with w ≡ 2 leaves the
/// VarCorr group variances unchanged and exactly doubles the residual
/// variance (not re-asserted here — this test only needs internal
/// consistency on a small synthetic LMM, cheaper than another R golden).
#[test]
fn fit_lmm_constant_weights_invariant() {
    let n_clusters = 6usize;
    let per = 8usize;
    let n = n_clusters * per;
    let mut st = 13u64;
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids_v = vec![0u32; n];
    for i in 0..n {
        ids_v[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        let re = 0.3 * ((ids_v[i] as f64) - (n_clusters as f64) / 2.0);
        y[i] = 0.5 + 0.4 * x1 + re + 0.2 * lcg(&mut st);
    }
    let model = ModelSpec {
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
    let unweighted = fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    let weighted = fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(vec![2.0; n]),
            ..FitOptions::default()
        },
    );
    assert!(unweighted.converged() && weighted.converged());
    // The θ̃=√c·θ substitution is exact algebra; the achieved match is
    // bounded by BOBYQA's rho_end floor (2 independently-converged fits,
    // not a shared trajectory), not by 1e-10 — 1e-6 relative is the tight
    // bound this floor actually supports (measured ~2e-8 on this fixture).
    for j in 0..2 {
        assert!(
            (unweighted.beta[j] - weighted.beta[j]).abs() / unweighted.beta[j].abs() < 1e-6,
            "β[{j}] unweighted {} vs w≡2 {}",
            unweighted.beta[j],
            weighted.beta[j]
        );
        assert!(
            (unweighted.se[j] - weighted.se[j]).abs() / unweighted.se[j] < 1e-6,
            "se[{j}] unweighted {} vs w≡2 {}",
            unweighted.se[j],
            weighted.se[j]
        );
    }
    assert_eq!(unweighted.tau2.len(), weighted.tau2.len());
    for k in 0..unweighted.tau2.len() {
        assert!(
            (unweighted.tau2[k] - weighted.tau2[k]).abs() / unweighted.tau2[k] < 1e-6,
            "tau2[{k}] unweighted {} vs w≡2 {}",
            unweighted.tau2[k],
            weighted.tau2[k]
        );
    }
}

/// Constant-weights invariance on a CROSSED random-slope design
/// (`y ~ 1 + x + (1 + x | g1) + (1 | g2)`, the `sim_slope` fixture):
/// w ≡ 2 must reproduce the unweighted β/SE/varcorr. This is the numeric
/// check for the crossed-path weight sites in `add_rows_multi` — the
/// intercept×intercept `zx += wᵢ` and the slope↔crossed `zx_slope += z·zw`
/// (q_p = 2 primary slope + crossed intercept extra takes the scalar
/// crossed branch, which unit-weight tests cannot distinguish from a
/// wrong-power bug). Same θ̃ = √c·θ rationale and BOBYQA-floor tolerance as
/// `fit_lmm_constant_weights_invariant`.
#[test]
fn fit_lmm_crossed_constant_weights_invariant() {
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
    let base_opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let unweighted = fit_cold(&x, &y, n, p, &model, &ids, &base_opts);
    let weighted = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            weights: Some(vec![2.0; n]),
            ..base_opts
        },
    );
    assert!(unweighted.converged() && weighted.converged());
    for j in 0..p {
        assert!(
            (unweighted.beta[j] - weighted.beta[j]).abs() / unweighted.beta[j].abs() < 1e-6,
            "β[{j}] unweighted {} vs w≡2 {}",
            unweighted.beta[j],
            weighted.beta[j]
        );
        assert!(
            (unweighted.se[j] - weighted.se[j]).abs() / unweighted.se[j] < 1e-6,
            "se[{j}] unweighted {} vs w≡2 {}",
            unweighted.se[j],
            weighted.se[j]
        );
    }
    // varcorr covers BOTH groupings' D̂ blocks (tau2 only reproduces the
    // (0,0) diagonal for the q=2 primary). Relative bound on the diagonals;
    // the small q=2 off-diagonal takes the same bound scaled to its own
    // magnitude floor.
    assert_eq!(unweighted.varcorr.len(), weighted.varcorr.len());
    for (gi, (vu, vw)) in unweighted
        .varcorr
        .iter()
        .zip(weighted.varcorr.iter())
        .enumerate()
    {
        assert_eq!(vu.len(), vw.len());
        for k in 0..vu.len() {
            let scale = vu[k].abs().max(1e-3);
            assert!(
                (vu[k] - vw[k]).abs() / scale < 1e-5,
                "varcorr[{gi}][{k}] unweighted {} vs w≡2 {}",
                vu[k],
                vw[k]
            );
        }
    }
}

/// Task 5 Step 6: the dense-LMM boundary (τ̂ ≈ 0, pinned exactly per the
/// Q7 deterministic-pin policy — mirrors
/// `lmm::tests::zero_between_cluster_variance_pins_at_exactly_zero`) must
/// reproduce the weighted fixed-only WLS fit (Task 1, `fit_ols`) on the
/// same rows: at θ̂=0 the mixed kernel's weighted Grams (`c`/`s`/`counts`,
/// all Σwᵢ-scaled per Task 5's accumulator) collapse to the same weighted
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
#[test]
fn fit_sim_slope_varcorr_is_pinned() {
    const REF_BETA: [f64; 2] = [1.038027232001732, 0.8009679266277173];
    const REF_SE: [f64; 2] = [0.33893198769052935, 0.17067326607075656];
    // varcorr[0] = g1, packed lower triangle [v00, c01, v11]; varcorr[1] = g2.
    const REF_VC_G1: [f64; 3] = [
        0.8994006042465587,
        -0.11776425179243186,
        0.39740694659663073,
    ];
    const REF_VC_G2: f64 = 0.5081866440363081;
    const REF_SIGMA2: f64 = 0.5717627834266172;

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
    let se_rel = (f.se[0] - REF_SE).abs() / REF_SE;
    assert!(
        se_rel < 2e-2,
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
    let se_rel = (f.se[0] - REF_SE).abs() / REF_SE;
    assert!(
        se_rel < 2e-2,
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

/// Bit-for-bit `LmmSweepOutcome` comparison — the arbiter for task 2's reuse
/// claim (a held `LmmSeamWs` must change nothing but wall time). `to_bits`
/// rather than `==` so a NaN-carrying `deviance`/`theta` (a non-converged
/// run) still compares meaningfully.
#[cfg(feature = "loop_advanced")]
fn assert_sweep_outcomes_bit_equal(a: &LmmSweepOutcome, b: &LmmSweepOutcome, label: &str) {
    assert_eq!(
        a.deviance.to_bits(),
        b.deviance.to_bits(),
        "{label}: deviance mismatch ({} vs {})",
        a.deviance,
        b.deviance
    );
    assert_eq!(
        a.theta.len(),
        b.theta.len(),
        "{label}: theta length mismatch"
    );
    for (i, (x, y)) in a.theta.iter().zip(&b.theta).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: theta[{i}] mismatch ({x} vs {y})"
        );
    }
    assert_eq!(a.n_eval, b.n_eval, "{label}: n_eval mismatch");
    assert_eq!(a.converged, b.converged, "{label}: converged mismatch");
}

/// Task 2's correctness proof: two [`lmm_sweep_fit_on`] calls at different
/// θ₀ on ONE [`build_lmm_seam_ws`] result must reproduce two independent
/// [`lmm_sweep_fit`] calls bit-for-bit — proving the held `suff`/`fit` (or
/// sparse `ws`) is genuinely reused, not silently rebuilt under the hood.
/// Dense (`Solver::NoZ`) shape: `lmm_hand_dataset`'s intercept-only design.
#[cfg(feature = "loop_advanced")]
#[test]
fn lmm_sweep_fit_on_matches_lmm_sweep_fit_dense() {
    let (x, y, n, p) = lmm_hand_dataset();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 6 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds::from_sizing(model.re.as_ref().unwrap(), n);
    assert!(matches!(classify_design(&model, 1), Solver::NoZ));

    let (mut ws, g) = build_lmm_seam_ws(&x, &y, n, p, &model, &ids);
    let (blind, lower, upper) = g.blind_theta_and_bounds();
    let theta_a = blind.clone();
    let theta_b: Vec<f64> = lower
        .iter()
        .zip(&upper)
        .map(|(&lo, &hi)| lo + 0.25 * (hi - lo))
        .collect();

    let on_a = lmm_sweep_fit_on(&mut ws, &g, Some(&theta_a), 1e-6, None, None);
    let on_b = lmm_sweep_fit_on(&mut ws, &g, Some(&theta_b), 1e-6, None, None);

    let standalone_a = lmm_sweep_fit(&x, &y, n, p, &model, &ids, Some(&theta_a), 1e-6, None, None);
    let standalone_b = lmm_sweep_fit(&x, &y, n, p, &model, &ids, Some(&theta_b), 1e-6, None, None);

    assert_sweep_outcomes_bit_equal(&on_a, &standalone_a, "dense theta_a");
    assert_sweep_outcomes_bit_equal(&on_b, &standalone_b, "dense theta_b");
}

/// Same proof as [`lmm_sweep_fit_on_matches_lmm_sweep_fit_dense`], sparse
/// (`Solver::Sparse`) shape: a crossed extra grouping carrying a slope
/// forces the sparse route regardless of size (`classify_design`'s
/// `slope_extras` clause), so a small hand design suffices.
#[cfg(feature = "loop_advanced")]
#[test]
fn lmm_sweep_fit_on_matches_lmm_sweep_fit_sparse() {
    let n = 32usize;
    let p = 3usize;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut primary = vec![0u32; n];
    let mut extra = vec![0u32; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        let x2 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = x2;
        primary[i] = (i % 4) as u32;
        extra[i] = ((i / 4) % 4) as u32;
        y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + 0.3 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                slopes: vec![1],
            }],
        }),
    };
    let ids = GroupIds {
        primary,
        extra: vec![extra],
    };
    assert!(matches!(classify_design(&model, 1), Solver::Sparse));

    let (mut ws, g) = build_lmm_seam_ws(&x, &y, n, p, &model, &ids);
    let (blind, lower, upper) = g.blind_theta_and_bounds();
    let theta_a = blind.clone();
    let theta_b: Vec<f64> = lower
        .iter()
        .zip(&upper)
        .map(|(&lo, &hi)| lo + 0.25 * (hi - lo))
        .collect();

    let on_a = lmm_sweep_fit_on(&mut ws, &g, Some(&theta_a), 1e-6, None, None);
    let on_b = lmm_sweep_fit_on(&mut ws, &g, Some(&theta_b), 1e-6, None, None);

    let standalone_a = lmm_sweep_fit(&x, &y, n, p, &model, &ids, Some(&theta_a), 1e-6, None, None);
    let standalone_b = lmm_sweep_fit(&x, &y, n, p, &model, &ids, Some(&theta_b), 1e-6, None, None);

    assert_sweep_outcomes_bit_equal(&on_a, &standalone_a, "sparse theta_a");
    assert_sweep_outcomes_bit_equal(&on_b, &standalone_b, "sparse theta_b");
}

/// [`lmm_objective_at`] self-consistency: evaluating it at the θ̂ an
/// `lmm_sweep_fit` run converged to must reproduce that run's own
/// `deviance` — both paths build the same `LmmSeamWs` and call the same
/// `reml_deviance`/`sparse_reml_deviance` closure, so only bit-level FP
/// order can separate them. Dense (`Solver::NoZ`) shape, same design as
/// [`lmm_sweep_fit_on_matches_lmm_sweep_fit_dense`].
#[cfg(feature = "loop_advanced")]
#[test]
fn lmm_objective_at_matches_lmm_sweep_fit_deviance_dense() {
    let (x, y, n, p) = lmm_hand_dataset();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 6 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds::from_sizing(model.re.as_ref().unwrap(), n);
    assert!(matches!(classify_design(&model, 1), Solver::NoZ));

    let outcome = lmm_sweep_fit(&x, &y, n, p, &model, &ids, None, 1e-6, None, None);
    assert!(outcome.converged, "dense sweep fit must converge");

    let obj = lmm_objective_at(&x, &y, n, p, &model, &ids, &outcome.theta);
    let rel = (obj - outcome.deviance).abs() / outcome.deviance.abs();
    assert!(
        rel < 1e-10,
        "dense: lmm_objective_at {obj} vs sweep deviance {} (rel {rel})",
        outcome.deviance
    );
}

/// Same proof as [`lmm_objective_at_matches_lmm_sweep_fit_deviance_dense`],
/// sparse (`Solver::Sparse`) shape: the crossed-slope 32-row design from
/// [`lmm_sweep_fit_on_matches_lmm_sweep_fit_sparse`] that forces
/// `classify_design` off the dense route.
#[cfg(feature = "loop_advanced")]
#[test]
fn lmm_objective_at_matches_lmm_sweep_fit_deviance_sparse() {
    let n = 32usize;
    let p = 3usize;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut primary = vec![0u32; n];
    let mut extra = vec![0u32; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        let x2 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = x2;
        primary[i] = (i % 4) as u32;
        extra[i] = ((i / 4) % 4) as u32;
        y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + 0.3 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                slopes: vec![1],
            }],
        }),
    };
    let ids = GroupIds {
        primary,
        extra: vec![extra],
    };
    assert!(matches!(classify_design(&model, 1), Solver::Sparse));

    let outcome = lmm_sweep_fit(&x, &y, n, p, &model, &ids, None, 1e-6, None, None);
    assert!(outcome.converged, "sparse sweep fit must converge");

    let obj = lmm_objective_at(&x, &y, n, p, &model, &ids, &outcome.theta);
    let rel = (obj - outcome.deviance).abs() / outcome.deviance.abs();
    assert!(
        rel < 1e-10,
        "sparse: lmm_objective_at {obj} vs sweep deviance {} (rel {rel})",
        outcome.deviance
    );
}

/// Task 3's correctness proof: two [`refit_lmm`] calls with DIFFERENT `y`
/// (dataset A unweighted, dataset B weighted) on ONE workspace built by
/// [`build_lmm_workspace`] must reproduce two independent [`fit_cold`]
/// calls bit-for-bit — proving genuine workspace reuse (not a silent
/// rebuild) AND exercising the `-Σlog wᵢ` weighted-deviance coupling: an
/// omitted correction would only surface as a `deviance` mismatch on
/// dataset B, since A (unweighted) can't distinguish the two code paths.
/// Dense (`Solver::NoZ`) shape: same intercept-only 6-cluster design as
/// `lmm_hand_dataset`, re-seeded per dataset.
#[cfg(feature = "loop_advanced")]
#[test]
fn refit_lmm_matches_fresh_fit_cold() {
    let n = 48usize;
    let p = 3usize;
    let n_clusters = 6usize;
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds::from_sizing(model.re.as_ref().unwrap(), n);

    // Two datasets of the SAME shape (n, p, cluster structure), different
    // y — mirrors MCPower's re-simulated-y power loop. Shape matches
    // `lmm_hand_dataset` (cluster c = i % n_clusters), parameterized by seed.
    let dataset = |seed: u64| -> (Vec<f64>, Vec<f64>) {
        let mut st = seed;
        let u_c: Vec<f64> = (0..n_clusters).map(|_| 0.6 * lcg(&mut st)).collect();
        let mut x = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let c = i % n_clusters;
            let x1 = lcg(&mut st);
            let x2 = lcg(&mut st);
            x[i * p] = 1.0;
            x[i * p + 1] = x1;
            x[i * p + 2] = x2;
            y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + u_c[c] + 0.8 * lcg(&mut st);
        }
        (x, y)
    };
    let (xa, ya) = dataset(42);
    let (xb, yb) = dataset(99);
    // Weighted case exercises the -Σlog wᵢ coupling: deviance is the field
    // that silently diverges from fit_cold's if the correction is dropped.
    let wb: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64 * 0.5).collect();

    let opts_a = FitOptions {
        target_indices: vec![1, 2],
        ..FitOptions::default()
    };
    let opts_b = FitOptions {
        target_indices: vec![1, 2],
        weights: Some(wb.clone()),
        ..FitOptions::default()
    };

    // ONE workspace, built once, reused across both refits — the reuse claim.
    let mut ws = build_lmm_workspace(p, &model, n);
    let refit_a = refit_lmm(&mut ws, &xa, &ya, n, p, &ids, &opts_a, None);
    let refit_b = refit_lmm(&mut ws, &xb, &yb, n, p, &ids, &opts_b, None);

    let cold_a = fit_cold(&xa, &ya, n, p, &model, &ids, &opts_a);
    let cold_b = fit_cold(&xb, &yb, n, p, &model, &ids, &opts_b);
    assert!(
        cold_a.converged() && cold_b.converged(),
        "oracle fits must converge"
    );

    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (label, refit, cold) in [
        ("A (unweighted)", &refit_a, &cold_a),
        ("B (weighted)", &refit_b, &cold_b),
    ] {
        assert_eq!(refit.converged(), cold.converged(), "{label}: converged");
        assert_eq!(bits(&refit.beta), bits(&cold.beta), "{label}: beta");
        assert_eq!(bits(&refit.se), bits(&cold.se), "{label}: se");
        assert_eq!(bits(&refit.tau2), bits(&cold.tau2), "{label}: tau2");
        assert_eq!(
            refit.varcorr.len(),
            cold.varcorr.len(),
            "{label}: varcorr len"
        );
        for (a, b) in refit.varcorr.iter().zip(&cold.varcorr) {
            assert_eq!(bits(a), bits(b), "{label}: varcorr block");
        }
        assert_eq!(
            refit.deviance.to_bits(),
            cold.deviance.to_bits(),
            "{label}: deviance"
        );
        assert_eq!(refit.n_eval, cold.n_eval, "{label}: n_eval");
        assert_eq!(refit.singular(), cold.singular(), "{label}: singular");
    }
}
