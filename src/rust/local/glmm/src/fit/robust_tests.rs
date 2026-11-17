// The assembly of a robust fit's pieces: built in the kernel's
// standardized coordinates and mapped to the caller's, they equal the kernel
// run on the caller's own design.

use super::super::common_tests::lmm_hand_dataset;
use crate::inference::cr2::{cr2_pieces, ClusterBlock};
use crate::inference::rlrt_refs::csv_columns;
use crate::inference::test_refs::{entries, load, Kind};
use crate::{Data, Family, FitOptions, GroupIds, ModelSpec, ReStructure, SeType, WeightPower};
use faer::linalg::solvers::Solve;
use faer::{Mat, MatRef};

/// One cluster's X_k, r_k, λ and U.
type Cluster = (Mat<f64>, Vec<f64>, Vec<f64>, Mat<f64>);

/// A random intercept and slope `(1 + x1 | g)` with weights: the fit's
/// robust `vcov` and `coef_df` against `cr2_pieces` on Φ_k = σ̂²/w +
/// Z_k D̂ Z_kᵀ with Z rows `[1, x1]` and X the caller's columns.
#[test]
fn slope_pieces_equal_the_kernel_on_the_callers_design() {
    let (x, y0, n, p) = lmm_hand_dataset();
    let y: Vec<f64> = (0..n)
        .map(|i| y0[i] + 0.3 * ((i % 6) as f64 - 2.5) * x[i * p + 1])
        .collect();
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    let ids = GroupIds::new((0..n).map(|i| (i % 6) as u32).collect());
    let spec = ModelSpec::mixed(Family::Gaussian, ReStructure::intercept().slopes([1]));
    let opts = FitOptions {
        weight_power: Some(WeightPower::Fixed(1.0)),
        se_type: Some(SeType::Robust),
        ..FitOptions::default()
    };
    let data = Data::new(&x, &y, n, p).ids(&ids).weights(&w);
    let fit = crate::fit(&data, &spec, &opts).unwrap();
    assert_ne!(fit.vcov, fit.model_vcov);
    let d = fit.varcorr(0);
    let l00 = d[0].sqrt();
    let l10 = d[2] / l00;
    let l11 = (d[3] - l10 * l10).sqrt();
    assert!(l11 > 0.0, "a slope variance to check against");
    let clusters: Vec<Vec<usize>> = (0..6).map(|c| (c..n).step_by(6).collect()).collect();
    let owned: Vec<Cluster> = clusters
        .iter()
        .map(|rows| {
            let xk = Mat::from_fn(rows.len(), p, |i, j| x[rows[i] * p + j]);
            let r = rows
                .iter()
                .map(|&i| y[i] - (0..p).map(|j| x[i * p + j] * fit.beta[j]).sum::<f64>())
                .collect();
            let lam = rows.iter().map(|&i| fit.dispersion / w[i]).collect();
            let u = Mat::from_fn(rows.len(), 2, |i, b| {
                let x1 = x[rows[i] * p + 1];
                if b == 0 {
                    l00 + x1 * l10
                } else {
                    x1 * l11
                }
            });
            (xk, r, lam, u)
        })
        .collect();
    let blocks: Vec<ClusterBlock<'_>> = owned
        .iter()
        .map(|(xk, r, lam, u)| ClusterBlock {
            x: xk.as_ref(),
            r,
            lambda: lam,
            u: u.as_ref(),
        })
        .collect();
    // R = Lᵀ, L the Cholesky factor of the model covariance's inverse.
    let m = MatRef::from_row_major_slice(&fit.model_vcov, p, p);
    let info = m.partial_piv_lu().solve(Mat::<f64>::identity(p, p));
    let r = crate::linalg::SeqLlt::new(info.as_ref())
        .unwrap()
        .L()
        .transpose()
        .to_owned();
    let pc = cr2_pieces(&blocks, r.as_ref()).unwrap();
    let v = pc.vcov();
    for i in 0..p {
        for j in 0..p {
            let got = fit.vcov[i * p + j];
            assert!(
                (got - v[(i, j)]).abs() <= 1e-10 * v[(i, i)].sqrt() * v[(j, j)].sqrt(),
                "vcov {i},{j}: {got} against {}",
                v[(i, j)]
            );
        }
        let mut e = vec![0.0; p];
        e[i] = 1.0;
        let df = pc.df(&e);
        assert!((fit.coef_df[i] / df - 1.0).abs() < 1e-10, "df {i}");
    }
}

/// An exact weighted fit (σ̂² = 0, no CR2 pieces) under the default δ fit
/// and under a robust fit at power 1: no panic, and a robust fit's SEs,
/// `vcov` and df are NaN, never the model's under the robust label. An
/// intercept on a constant y with weights summing to 16: the Cholesky
/// factor (4), β̂ = 4 and every residual are exact.
#[test]
fn an_exact_fit_has_nan_robust_ses() {
    let n = 4;
    let x = vec![1.0; n];
    let y = vec![4.0; n];
    let w = vec![1.0, 3.0, 4.0, 8.0];
    let data = Data::new(&x, &y, n, 1).weights(&w);
    let spec = ModelSpec::fixed(Family::Gaussian);
    let robust = FitOptions {
        weight_power: Some(WeightPower::Fixed(1.0)),
        se_type: Some(SeType::Robust),
        ..FitOptions::default()
    };
    // The δ fit lands on a tiny positive σ̂², so it has pieces.
    let _ = crate::fit(&data, &spec, &FitOptions::default());
    let fit = crate::fit(&data, &spec, &robust).unwrap();
    assert_eq!(fit.dispersion(), 0.0);
    assert_eq!(fit.se_type, SeType::Robust);
    assert!(fit.robust.is_none());
    assert!(fit.se.iter().all(|v| v.is_nan()), "{:?}", fit.se);
    assert!(fit.vcov.iter().all(|v| v.is_nan()));
    assert!(fit.coef_df.iter().all(|v| v.is_nan()));
}

/// CR2 on y·c for c far from 1: the SEs scale by c and the df do not move,
/// or, where σ̂² leaves f64's range, NaN without a panic.
#[test]
fn robust_ses_scale_with_y() {
    let (x, y, n, p) = lmm_hand_dataset();
    let ids = GroupIds::new((0..n).map(|i| (i % 6) as u32).collect());
    let spec = ModelSpec::mixed(Family::Gaussian, ReStructure::intercept());
    let opts = FitOptions {
        se_type: Some(SeType::Robust),
        ..FitOptions::default()
    };
    let fit_at = |c: f64| {
        let yc: Vec<f64> = y.iter().map(|v| v * c).collect();
        crate::fit(&Data::new(&x, &yc, n, p).ids(&ids), &spec, &opts).unwrap()
    };
    let base = fit_at(1.0);
    for c in [1e-100, 1e99, 1e-170] {
        let fit = fit_at(c);
        if fit.robust.is_none() {
            assert!(fit.se.iter().all(|v| v.is_nan()), "{c}: {:?}", fit.se);
            continue;
        }
        for j in 0..p {
            let (se, df) = (fit.se[j] / c, fit.coef_df[j]);
            assert!(
                (se / base.se[j] - 1.0).abs() < 1e-6,
                "{c}: se {j} {se} against {}",
                base.se[j]
            );
            assert!((df / base.coef_df[j] - 1.0).abs() < 1e-6, "{c}: df {j}");
        }
    }
    for c in [1e-100, 1e99] {
        assert!(fit_at(c).robust.is_some(), "{c}: pieces in range");
    }
}

/// A treatment-coded factor whose reference level `a` occurs in cluster 0
/// only: its two columns are each identified by that cluster alone (NaN), but
/// b − c is not, and its robust test equals the c column's of the same model
/// coded with b as the reference.
#[test]
fn an_identified_contrast_of_unidentified_columns_is_tested() {
    let (x0, y, n, _) = lmm_hand_dataset();
    let level = |i: usize| match (i % 6, (i / 6) % 2) {
        (0, 0) => 0,
        (_, 0) => 1,
        _ => 2,
    };
    let coded = |reference: usize| -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                let others = (0..3).filter(move |&l| l != reference);
                std::iter::once(1.0)
                    .chain(others.map(move |l| f64::from(u8::from(level(i) == l))))
                    .chain(std::iter::once(x0[i * 3 + 1]))
            })
            .collect()
    };
    let ids = GroupIds::new((0..n).map(|i| (i % 6) as u32).collect());
    let spec = ModelSpec::mixed(Family::Gaussian, ReStructure::intercept());
    let opts = FitOptions {
        se_type: Some(SeType::Robust),
        ..FitOptions::default()
    };
    let (xa, xb) = (coded(0), coded(1));
    let fa = crate::fit(&Data::new(&xa, &y, n, 4).ids(&ids), &spec, &opts).unwrap();
    let fb = crate::fit(&Data::new(&xb, &y, n, 4).ids(&ids), &spec, &opts).unwrap();
    assert!(fa.se[1].is_nan() && fa.se[2].is_nan(), "{:?}", fa.se);
    let t = fa.contrast_t(&[0.0, -1.0, 1.0, 0.0]).unwrap();
    assert!(t.notes.is_empty(), "{:?}", t.notes);
    let want = fb.contrast_t(&[0.0, 0.0, 1.0, 0.0]).unwrap();
    assert!(
        (t.se / want.se - 1.0).abs() < 1e-6,
        "{} against {}",
        t.se,
        want.se
    );
    assert!(
        (t.df / want.df - 1.0).abs() < 1e-6,
        "{} against {}",
        t.df,
        want.df
    );
}

/// The HC2 entry of the `cr2_df` references (OLS, two columns 1e-7 apart)
/// through the fit, with its own factor: each `coef_df` against its 50-digit
/// df.
#[test]
fn hc2_df_match_50_digits_on_a_nearly_collinear_design() {
    let list: Vec<_> = entries(Kind::Cr2Df)
        .into_iter()
        .filter(|e| e.raw.get("cluster").is_none())
        .collect();
    assert_eq!(list.len(), 1, "cr2_df entries without clusters");
    let e = &list[0];
    let d = csv_columns(e.data.as_deref().unwrap());
    let n = d["y"].len();
    let x: Vec<f64> = (0..n)
        .flat_map(|i| [1.0, d["a"][i], d["xi"][i], d["xi2"][i]])
        .collect();
    let opts = FitOptions {
        se_type: Some(SeType::Robust),
        ..FitOptions::default()
    };
    let spec = ModelSpec::fixed(Family::Gaussian);
    let fit = crate::fit(&Data::new(&x, &d["y"], n, 4), &spec, &opts).unwrap();
    let want: Vec<f64> = load(Kind::Cr2Df, &e.name)["df"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!((fit.coef_df.len(), want.len()), (4, 4));
    for (j, (got, want)) in fit.coef_df.iter().zip(&want).enumerate() {
        let rel = (got / want - 1.0).abs();
        assert!(rel <= 1e-8, "df {j}: {got} against {want}: {rel:.2e}");
    }
}
