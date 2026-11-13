//! `predict` on the matrix door: the training rows against `Fit::fitted`, the
//! population predictor, NaN coefficients, unseen levels, the standard error
//! against a dense joint inverse, and the refused inputs.

use super::*;
use crate::formula::{self, Lowered};
use crate::test_support::csv_table;
use crate::{BinomialLink, Data, FitOptions, GammaLink, GroupIds, ModelSpec, PoissonLink, Re};

/// The in-crate band between two crate paths, relative to `1 + Σ|term|`:
/// `fitted` runs through the standardized kernel, `predict` through the
/// caller's design.
const ROUNDOFF: f64 = 1e-10;

pub(crate) const SLEEP: &str = include_str!("../../validation/data/empirical/sleepstudy.csv");
pub(crate) const BERN: &str =
    include_str!("../../validation/data/simulated/sim_factorial_bern.csv");

pub(crate) fn binomial() -> Family {
    Family::Binomial {
        link: BinomialLink::Logit,
    }
}

pub(crate) fn lowered_fit(
    csv: &str,
    factors: &[&str],
    formula: &str,
    family: Family,
) -> (Lowered, Fit) {
    let lo = formula::lower(formula, &csv_table(csv, factors), family).unwrap();
    let opts = FitOptions {
        condvar: true,
        ..FitOptions::default()
    };
    let fit = crate::fit(&lo.data(), lo.spec(), &opts).unwrap();
    assert!(fit.converged(), "{formula}: not converged");
    (lo, fit)
}

/// The terms of row `i`'s conditional linear predictor on the training design:
/// the offset, each `x_j β̂_j`, and each `z_r b̂_r` of the row's levels, read
/// off `Fit::ranef` with the reported columns of each grouping.
fn row_terms(lo: &Lowered, fit: &Fit, i: usize) -> Vec<f64> {
    let p = lo.p;
    let mut t: Vec<f64> = (0..p).map(|j| lo.x[i * p + j] * fit.beta[j]).collect();
    t.push(lo.offset.as_ref().map_or(0.0, |o| o[i]));
    let Some(re) = lo.spec().re.as_ref() else {
        return t;
    };
    let col = |c: usize| {
        if c < p {
            lo.x[i * p + c]
        } else {
            lo.re_columns[i * lo.re_p + c - p]
        }
    };
    let groupings = std::iter::once((&re.slopes, &re.terms, &lo.ids.primary)).chain(
        re.extra_groupings
            .iter()
            .zip(&lo.ids.extra)
            .map(|(g, e)| (&g.slopes, &g.terms, e)),
    );
    let mut at = 0;
    for (g, (slopes, terms, ids)) in groupings.enumerate() {
        let hidden = !terms.is_empty() && !terms.iter().any(|t| t.intercept);
        let mut z: Vec<f64> = if hidden { vec![] } else { vec![1.0] };
        z.extend(slopes.iter().map(|&c| col(c as usize)));
        let base = at + ids[i] as usize * z.len();
        t.extend(z.iter().enumerate().map(|(r, v)| v * fit.ranef[base + r]));
        at += fit.ranef_levels[g] * z.len();
    }
    t
}

fn band(terms: &[f64]) -> f64 {
    ROUNDOFF * (1.0 + terms.iter().map(|t| t.abs()).sum::<f64>())
}

/// The ten shapes of the training-row checks: OLS, a binomial GLM, the LMM
/// routes (per-group with a correlated, an uncorrelated and an intercept-free
/// block and a random-effect-only slope, nested, crossed), a binomial GLMM and
/// a Poisson GLMM with an offset.
fn ten_shapes() -> Vec<(Lowered, Fit, &'static str)> {
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let shapes: [(&str, &[&str], &str, Family); 10] = [
        (
            include_str!("../../validation/data/simulated/sim_factorial.csv"),
            &["subj", "g", "h"],
            "y ~ x + g",
            Family::Gaussian,
        ),
        (BERN, &["subj", "g", "h"], "y ~ x + g", binomial()),
        (
            SLEEP,
            &["Subject"],
            "Reaction ~ Days + (1 + Days | Subject)",
            Family::Gaussian,
        ),
        (
            SLEEP,
            &["Subject"],
            "Reaction ~ Days + (Days || Subject)",
            Family::Gaussian,
        ),
        (
            SLEEP,
            &["Subject"],
            "Reaction ~ Days + (0 + Days | Subject)",
            Family::Gaussian,
        ),
        (
            SLEEP,
            &["Subject"],
            "Reaction ~ 1 + (Days | Subject)",
            Family::Gaussian,
        ),
        (
            include_str!("../../validation/data/empirical/Pastes.csv"),
            &["batch", "cask", "sample"],
            "strength ~ 1 + (1 | batch/cask)",
            Family::Gaussian,
        ),
        (
            include_str!("../../validation/data/empirical/Penicillin.csv"),
            &["plate", "sample"],
            "diameter ~ 1 + (1 | plate) + (1 | sample)",
            Family::Gaussian,
        ),
        (
            BERN,
            &["subj", "g", "h"],
            "y ~ x + g + (1 | subj)",
            binomial(),
        ),
        (
            include_str!("../../validation/data/simulated/sim_poisson_offset.csv"),
            &["cluster"],
            "y ~ x + offset(log_exposure) + (1 | cluster)",
            poisson,
        ),
    ];
    shapes
        .into_iter()
        .map(|(csv, factors, formula, family)| {
            let (lo, fit) = lowered_fit(csv, factors, formula, family);
            (lo, fit, formula)
        })
        .collect()
}

/// Conditional, response scale, on the training rows: `Fit::fitted`. Population
/// mode on the link scale is `Σ x_j β̂_j + o`, exactly.
#[test]
fn predict_on_the_training_rows_equals_fitted() {
    let opts = PredictOptions {
        scale: Scale::Response,
        ..PredictOptions::default()
    };
    let population = PredictOptions {
        population: true,
        ..PredictOptions::default()
    };
    let mut ran = 0;
    for (lo, fit, formula) in ten_shapes() {
        let pred = predict(&fit, lo.spec(), &lo.data().design, &opts).unwrap();
        assert_eq!(pred.fit.len(), lo.n, "{formula}");
        assert!(pred.notes.is_empty(), "{formula}: {:?}", pred.notes);
        for i in 0..lo.n {
            let gap = (pred.fit[i] - fit.fitted[i]).abs();
            let tol = band(&row_terms(&lo, &fit, i));
            assert!(gap <= tol, "{formula} row {i}: gap {gap:e} over {tol:e}");
        }
        let pop = predict(&fit, lo.spec(), &lo.data().design, &population).unwrap();
        assert!(pop.notes.is_empty(), "{formula}: {:?}", pop.notes);
        let p = lo.p;
        for i in 0..lo.n {
            let xb: f64 = (0..p).map(|j| lo.x[i * p + j] * fit.beta[j]).sum();
            let want = xb + lo.offset.as_ref().map_or(0.0, |o| o[i]);
            assert_eq!(pop.fit[i].to_bits(), want.to_bits(), "{formula} row {i}");
        }
        ran += 1;
    }
    assert_eq!(ran, 10);
}

/// A NaN coefficient contributes zero: an aliased column gives the
/// prediction, SE and interval of the design without it, and a GLM column the
/// fit could not separate gives the prediction with its coefficient at zero
/// and a NaN SE (its `vcov` is NaN).
#[test]
fn nan_coefficient_contributes_zero() {
    let lo = formula::lower(
        "Reaction ~ Days + (1 | Subject)",
        &csv_table(SLEEP, &["Subject"]),
        Family::Gaussian,
    )
    .unwrap();
    let n = lo.n;
    let x2 = lo.x.clone();
    let x3: Vec<f64> = x2
        .chunks(2)
        .flat_map(|r| [r[0], r[1], 2.0 * r[1]])
        .collect();
    let spec = ModelSpec::mixed(Family::Gaussian, Re::intercept());
    let opts = FitOptions {
        condvar: true,
        ..FitOptions::default()
    };
    let fit2 = crate::fit(&Data::new(&x2, &lo.y, n, 2).ids(&lo.ids), &spec, &opts).unwrap();
    let fit3 = crate::fit(&Data::new(&x3, &lo.y, n, 3).ids(&lo.ids), &spec, &opts).unwrap();
    assert!(fit3.beta[2].is_nan() && fit3.diagnostics.aliased[2]);
    let popts = PredictOptions {
        se: true,
        ..PredictOptions::default()
    };
    let p2 = predict(&fit2, &spec, &Design::new(&x2, n, 2).ids(&lo.ids), &popts).unwrap();
    let p3 = predict(&fit3, &spec, &Design::new(&x3, n, 3).ids(&lo.ids), &popts).unwrap();
    for (what, a, b) in [
        ("fit", &p2.fit, &p3.fit),
        ("se", &p2.se, &p3.se),
        ("lower", &p2.lower, &p3.lower),
        ("upper", &p2.upper, &p3.upper),
    ] {
        for i in 0..n {
            let tol = ROUNDOFF * (1.0 + a[i].abs());
            assert!((a[i] - b[i]).abs() <= tol, "aliased {what} row {i}");
        }
    }

    // The breakdown design of the GLM conditioning test: column 3 duplicates
    // column 2, through the loop tier so no standardization separates them.
    let (n, p) = (50usize, 4usize);
    let mut x = Vec::with_capacity(n * p);
    for i in 0..n {
        let e = f64::from(u8::from(i == n - 1));
        x.extend_from_slice(&[1.0, i as f64 / n as f64 - 0.5, e, e]);
    }
    let y: Vec<f64> = (0..n)
        .map(|i| {
            if i == n - 1 {
                3.0
            } else {
                ((7 * i) % 5) as f64
            }
        })
        .collect();
    let model = ModelSpec::fixed(Family::Poisson {
        link: PoissonLink::Log,
    });
    let mut ws = crate::fit::build_workspace(&model, &GroupIds::default(), n, p, &opts);
    ws.set_prestandardized(true);
    let data = Data::new(&x, &y, n, p);
    let view = crate::fit::fit_on(&mut ws, &data, None, &opts);
    let fit = view.into_fit(&data, &model, &opts);
    assert!(fit.beta[3].is_nan() && fit.beta[..3].iter().all(|b| b.is_finite()));
    let new = Design::new(&x, n, p);
    let got = predict(&fit, &model, &new, &popts).unwrap();
    let mut zeroed = fit.beta.clone();
    zeroed[3] = 0.0;
    let parts = FitParts {
        beta: &zeroed,
        ..FitParts::of(&fit, &model)
    };
    let want = predict_parts(&parts, &model, &new, &PredictOptions::default()).unwrap();
    for i in 0..n {
        assert_eq!(got.fit[i].to_bits(), want.fit[i].to_bits(), "G3 row {i}");
    }
    assert!(got.se.iter().all(|s| s.is_nan()), "G3 SE");
    assert!(got.lower.iter().chain(&got.upper).all(|v| v.is_nan()));
}

/// sleepstudy's `Reaction ~ Days + (1 | Subject)` on `[1, Days]` and on
/// `[1, Days, 2·Days]`, whose third column is aliased: the lowering, the
/// three-column design, and the two fits with condvar on.
pub(crate) fn doubled_days_fits() -> (Lowered, Vec<f64>, Fit, Fit) {
    let lo = formula::lower(
        "Reaction ~ Days + (1 | Subject)",
        &csv_table(SLEEP, &["Subject"]),
        Family::Gaussian,
    )
    .unwrap();
    let x3: Vec<f64> =
        lo.x.chunks(2)
            .flat_map(|r| [r[0], r[1], 2.0 * r[1]])
            .collect();
    let spec = ModelSpec::mixed(Family::Gaussian, Re::intercept());
    let opts = FitOptions {
        condvar: true,
        ..FitOptions::default()
    };
    let n = lo.n;
    let fit2 = crate::fit(&Data::new(&lo.x, &lo.y, n, 2).ids(&lo.ids), &spec, &opts).unwrap();
    let fit3 = crate::fit(&Data::new(&x3, &lo.y, n, 3).ids(&lo.ids), &spec, &opts).unwrap();
    assert!(fit3.diagnostics.aliased[2]);
    (lo, x3, fit2, fit3)
}

/// A rank-deficient fit predicts its own training rows as it fitted them, a
/// new row inside the row space (`x2 = 2x`) from the kept coefficients, and a
/// row outside it as NaN with a note; an all-zero row stays estimable.
#[test]
fn a_row_outside_the_row_space_of_a_rank_deficient_fit_predicts_nan() {
    let (lo, x3, fit2, fit3) = doubled_days_fits();
    let spec = ModelSpec::mixed(Family::Gaussian, Re::intercept());
    let popts = PredictOptions {
        se: true,
        ..PredictOptions::default()
    };
    let train = predict(
        &fit3,
        &spec,
        &Design::new(&x3, lo.n, 3).ids(&lo.ids),
        &popts,
    )
    .unwrap();
    assert!(train.notes.is_empty(), "{:?}", train.notes);
    for part in [&train.fit, &train.se, &train.lower, &train.upper] {
        assert!(part.iter().all(|v| v.is_finite()));
    }
    for (i, (&got, &want)) in train.fit.iter().zip(&fit3.fitted).enumerate() {
        assert!(
            (got - want).abs() <= ROUNDOFF * (1.0 + want.abs()),
            "row {i}"
        );
    }

    let pop = PredictOptions {
        population: true,
        ..popts
    };
    let rows = [1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0];
    let got = predict(&fit3, &spec, &Design::new(&rows, 3, 3), &pop).unwrap();
    let want = predict(&fit2, &spec, &Design::new(&[1.0, 1.0], 1, 2), &pop).unwrap();
    for (what, a, b) in [("fit", &got.fit, &want.fit), ("se", &got.se, &want.se)] {
        assert!(
            (a[0] - b[0]).abs() <= ROUNDOFF * (1.0 + b[0].abs()),
            "{what}"
        );
    }
    for part in [&got.fit, &got.se, &got.lower, &got.upper] {
        assert!(part[1].is_nan() && part[2].is_finite());
    }
    assert!(
        matches!(&got.notes[..], [Note::RowsNotEstimable { rows }] if rows == &[1]),
        "{:?}",
        got.notes
    );
}

/// 36 rows of counts `y` over the cells a:B, b:A and b:B of `f1` (a, b) by
/// `f2` (A, B), with six levels of `g` crossed with the cells; the cell a:A
/// has no row.
pub(crate) fn absent_cell_csv() -> String {
    let cells = [("a", "B"), ("b", "A"), ("b", "B")];
    let mut csv = String::from("y,f1,f2,g\n");
    for i in 0..36 {
        let (f1, f2) = cells[i % 3];
        let y = 1 + (7 * i) % 5 + 2 * (i % 3);
        csv.push_str(&format!("{y},{f1},{f2},g{}\n", (i / 3) % 6));
    }
    csv
}

/// The cell-means fit `y ~ 0 + f1:f2 + (1 | g)` with an absent cell: the
/// cell's column is aliased, its row is NaN in every part with a note, and a
/// cell with data predicts a number.
#[test]
fn an_absent_cell_of_a_cell_means_fit_predicts_nan() {
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let (lo, fit) = lowered_fit(
        &absent_cell_csv(),
        &["f1", "f2", "g"],
        "y ~ 0 + f1:f2 + (1 | g)",
        poisson,
    );
    let col = |name: &str| lo.col_names.iter().position(|c| c == name).unwrap();
    let (aa, bb) = (col("f1a:f2A"), col("f1b:f2B"));
    assert!(fit.diagnostics.aliased[aa]);
    let p = lo.p;
    let mut rows = vec![0.0; 2 * p];
    rows[aa] = 1.0;
    rows[p + bb] = 1.0;
    let opts = PredictOptions {
        population: true,
        se: true,
        future: true,
        ..PredictOptions::default()
    };
    let got = predict(&fit, lo.spec(), &Design::new(&rows, 2, p), &opts).unwrap();
    for part in [
        &got.fit,
        &got.se,
        &got.lower,
        &got.upper,
        &got.future_lower,
        &got.future_upper,
    ] {
        assert!(part[0].is_nan() && part[1].is_finite());
    }
    assert!(
        matches!(&got.notes[..], [Note::RowsNotEstimable { rows }] if rows == &[0]),
        "{:?}",
        got.notes
    );
}

/// A fit that returned no estimates (every coefficient NaN, nothing aliased)
/// predicts NaN with a NaN standard error, interval and future bounds, on
/// the link and the response scale.
#[test]
fn a_fit_without_estimates_predicts_nan() {
    // n = p = 2: OLS and the Poisson GLM end not converged with no estimates.
    let x = [1.0, 0.3, 1.0, -0.4];
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    for (family, y) in [(Family::Gaussian, [0.5, 1.5]), (poisson, [1.0, 2.0])] {
        let spec = ModelSpec::fixed(family);
        let fit = crate::fit(&Data::new(&x, &y, 2, 2), &spec, &FitOptions::default()).unwrap();
        assert!(!fit.converged() && fit.beta.iter().all(|b| b.is_nan()));
        assert!(fit.diagnostics.aliased.iter().all(|a| !a));
        let new = Design::new(&x, 2, 2);
        for scale in [Scale::Link, Scale::Response] {
            let opts = PredictOptions {
                se: true,
                scale,
                ..PredictOptions::default()
            };
            let got = predict(&fit, &spec, &new, &opts).unwrap();
            for v in [&got.fit, &got.se, &got.lower, &got.upper] {
                assert_eq!(v.len(), 2);
                assert!(v.iter().all(|e| e.is_nan()), "{family:?} {scale:?}: {v:?}");
            }
        }
    }

    // A Poisson GLMM on the degenerate route (n = p = 3), in population mode
    // with future bounds.
    let (n, p) = (3usize, 3usize);
    let x: Vec<f64> = (0..n * p).map(|k| (k % 5) as f64 + 0.5).collect();
    let y = vec![1.0, 2.0, 0.0];
    let ids = GroupIds {
        primary: vec![0, 1, 1],
        extra: vec![],
    };
    let spec = ModelSpec {
        family: poisson,
        re: Some(crate::ReStructure {
            terms: vec![crate::ReTerm::no_intercept().slopes([1u32])],
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let fit = crate::fit(
        &Data::new(&x, &y, n, p).ids(&ids),
        &spec,
        &FitOptions::default(),
    )
    .unwrap();
    assert!(!fit.converged() && fit.beta.iter().all(|b| b.is_nan()));
    let opts = PredictOptions {
        population: true,
        future: true,
        se: true,
        ..PredictOptions::default()
    };
    let got = predict(&fit, &spec, &Design::new(&x, n, p), &opts).unwrap();
    for v in [&got.fit, &got.se, &got.future_lower, &got.future_upper] {
        assert_eq!(v.len(), n);
        assert!(v.iter().all(|e| e.is_nan()), "GLMM future: {v:?}");
    }
}

/// On Penicillin, ids at or past `Fit::ranef_levels` in one or both groupings
/// take a zero effect for that grouping only, and raise one note per grouping
/// with the exact rows; refused when new levels are not allowed.
#[test]
fn unseen_ids_contribute_zero_and_raise_one_note_per_grouping() {
    let (lo, fit) = lowered_fit(
        include_str!("../../validation/data/empirical/Penicillin.csv"),
        &["plate", "sample"],
        "diameter ~ 1 + (1 | plate) + (1 | sample)",
        Family::Gaussian,
    );
    let (k0, k1) = (fit.ranef_levels[0], fit.ranef_levels[1]);
    let rows = 6;
    let mut g0: Vec<u32> = lo.ids.primary[..rows].to_vec();
    let mut g1: Vec<u32> = lo.ids.extra[0][..rows].to_vec();
    g0[1] = k0 as u32;
    g1[2] = k1 as u32 + 3;
    g0[3] = k0 as u32 + 1;
    g1[3] = k1 as u32;
    let ids = GroupIds::new(g0.clone()).extra(g1.clone());
    let new = Design::new(&lo.x[..rows], rows, 1).ids(&ids);
    let pred = predict(&fit, lo.spec(), &new, &PredictOptions::default()).unwrap();
    for i in 0..rows {
        let b0 = (g0[i] < k0 as u32).then(|| fit.ranef[g0[i] as usize]);
        let b1 = (g1[i] < k1 as u32).then(|| fit.ranef[k0 + g1[i] as usize]);
        let want = fit.beta[0] + b0.unwrap_or(0.0) + b1.unwrap_or(0.0);
        let terms = [fit.beta[0], b0.unwrap_or(0.0), b1.unwrap_or(0.0)];
        assert!((pred.fit[i] - want).abs() <= band(&terms), "row {i}");
    }
    assert!(
        matches!(
            &pred.notes[..],
            [
                Note::NewGroupingLevels { grouping: 0, rows: r0 },
                Note::NewGroupingLevels { grouping: 1, rows: r1 },
            ] if r0 == &[1, 3] && r1 == &[2, 3]
        ),
        "{:?}",
        pred.notes
    );
    let strict = PredictOptions {
        allow_new_levels: false,
        ..PredictOptions::default()
    };
    let Err(FitError::Ids(m)) = predict(&fit, lo.spec(), &new, &strict) else {
        panic!("an unseen level is refused without allow_new_levels");
    };
    assert!(m.contains("grouping 0") && m.contains("row 1"), "{m}");
    assert_eq!(m.row(), Some(1));
    assert!(m.text_with_row(6).contains("at row 6 (id "), "{m}");
}

/// The conditional SE on sleepstudy equals lme4's `vcov_full` form,
/// `σ̂·sqrt([Λᵀz; x]ᵀM⁻¹[Λᵀz; x])` with `M` the dense joint mixed-model matrix
/// `[[ΛᵀZᵀZΛ + I, ΛᵀZᵀX], [XᵀZΛ, XᵀX]]`; and the response-scale interval of a
/// Gamma inverse-link GLMM has its ends ordered.
#[test]
fn se_and_interval_follow_their_formulas() {
    use faer::linalg::solvers::Solve;
    use faer::Mat;
    let (lo, fit) = lowered_fit(
        SLEEP,
        &["Subject"],
        "Reaction ~ Days + (1 + Days | Subject)",
        Family::Gaussian,
    );
    let (n, p, levels) = (lo.n, lo.p, fit.ranef_levels[0]);
    let sigma_sq = fit.dispersion;
    // Λ: the lower factor of varcorr / σ̂² (vech (0,0), (1,0), (1,1)).
    let v = &fit.varcorr[0];
    let l00 = (v[0] / sigma_sq).sqrt();
    let l10 = v[1] / sigma_sq / l00;
    let l11 = (v[2] / sigma_sq - l10 * l10).sqrt();
    let kt = 2 * levels;
    // Row i of [ZΛ | X].
    let joint = Mat::<f64>::from_fn(n, kt + p, |i, k| {
        let days = lo.x[i * p + 1];
        let lvl = lo.ids.primary[i] as usize;
        if k >= kt {
            lo.x[i * p + k - kt]
        } else if k / 2 != lvl {
            0.0
        } else if k % 2 == 0 {
            l00 + days * l10
        } else {
            days * l11
        }
    });
    let mut m = joint.transpose() * &joint;
    for k in 0..kt {
        m[(k, k)] += 1.0;
    }
    let sol = m.partial_piv_lu().solve(joint.transpose());
    let opts = PredictOptions {
        se: true,
        ..PredictOptions::default()
    };
    let pred = predict(&fit, lo.spec(), &lo.data().design, &opts).unwrap();
    for i in 0..n {
        let q: f64 = (0..kt + p).map(|k| joint[(i, k)] * sol[(k, i)]).sum();
        let want = (sigma_sq * q).sqrt();
        let tol = ROUNDOFF * (1.0 + want);
        assert!(
            (pred.se[i] - want).abs() <= tol,
            "row {i}: {} vs {want}",
            pred.se[i]
        );
    }

    let (lo, fit) = lowered_fit(
        include_str!("../../validation/data/simulated/sim_gamma.csv"),
        &["cluster", "grp"],
        "y ~ x + (1 | cluster)",
        Family::Gamma {
            link: GammaLink::Inverse,
        },
    );
    let opts = PredictOptions {
        se: true,
        scale: Scale::Response,
        ..PredictOptions::default()
    };
    let pred = predict(&fit, lo.spec(), &lo.data().design, &opts).unwrap();
    for i in 0..lo.n {
        assert!(
            pred.lower[i] < pred.fit[i] && pred.fit[i] < pred.upper[i],
            "row {i}: [{}, {}] around {}",
            pred.lower[i],
            pred.upper[i],
            pred.fit[i]
        );
    }
}

/// On the response scale of a Gamma inverse-link fit, a row with η̂ ≤ 0 has no
/// mean: its fit, SE, interval and future bounds are NaN. A row with η̂ > 0
/// whose interval reaches η ≤ 0 keeps a finite fit and lower end, and its
/// upper end is +∞.
#[test]
fn a_reciprocal_link_row_with_eta_at_or_below_zero_predicts_nan() {
    let (lo, fit) = lowered_fit(
        include_str!("../../validation/data/simulated/sim_gamma.csv"),
        &["cluster", "grp"],
        "y ~ x + (1 | cluster)",
        Family::Gamma {
            link: GammaLink::Inverse,
        },
    );
    let (b0, b1) = (fit.beta[0], fit.beta[1]);
    assert!(b0 > 0.0 && b1 != 0.0, "b0 {b0}, b1 {b1}");
    // On the design [1, x], η̂ = t·b0 at x = (t − 1)·b0/b1.
    let x_at = |t: f64| (t - 1.0) * b0 / b1;
    let cands: Vec<f64> = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001]
        .iter()
        .flat_map(|&t| [1.0, x_at(t)])
        .collect();
    let link_opts = PredictOptions {
        population: true,
        se: true,
        ..PredictOptions::default()
    };
    let link = predict(&fit, lo.spec(), &Design::new(&cands, 7, 2), &link_opts).unwrap();
    let k = (0..7)
        .find(|&k| link.fit[k] - 1.96 * link.se[k] <= 0.0)
        .expect("a candidate whose interval reaches η ≤ 0");
    let eta_b = link.fit[k];
    let rows = [1.0, x_at(-0.5), 1.0, cands[2 * k + 1]];
    let opts = PredictOptions {
        population: true,
        se: true,
        future: true,
        scale: Scale::Response,
        ..PredictOptions::default()
    };
    let got = predict(&fit, lo.spec(), &Design::new(&rows, 2, 2), &opts).unwrap();
    for v in [
        &got.fit,
        &got.se,
        &got.lower,
        &got.upper,
        &got.future_lower,
        &got.future_upper,
    ] {
        assert!(v[0].is_nan(), "row with η̂ ≤ 0: {v:?}");
    }
    assert!(got.fit[1].is_finite() && got.fit[1] == 1.0 / eta_b);
    assert!(got.lower[1].is_finite() && got.lower[1] > 0.0, "{got:?}");
    assert_eq!(got.upper[1], f64::INFINITY);
    assert!(got.future_lower[1].is_nan() && got.future_upper[1].is_nan());
}

/// The conditional SE equals `sqrt(zᵀCz + dᵀVd)`, `d = x − Kᵀz`, with `C` and
/// `K` the dense inverse over the whole random-effect vector
/// (`common_tests::dense_joint`) and `V` = `Fit::vcov`, so the covariance
/// between levels of different groupings is in: on the training rows, and on
/// rows where one grouping's level is unseen (its columns of `z` zero) or a
/// nested level is moved under another parent than the row's. Shapes:
/// a dense crossed tail (Penicillin), a nested family (Pastes), both together,
/// per-group fits with a correlated, an uncorrelated block and a
/// random-effect-only slope, and on the GLMM structured layout a nested family
/// with a crossed tail (cloglog) and Poisson fits with two and three crossed
/// groupings in orders the size rule keeps and swaps, and a slope grouping
/// written last with the first grouping nested in it.
#[test]
fn conditional_se_matches_a_dense_joint_inverse() {
    let mix_f: &[&str] = &["g1", "g2", "c1"];
    let grouse = include_str!("../../validation/data/empirical/grouseticks.csv");
    let grouse_f: &[&str] = &["INDEX", "BROOD", "YEAR", "LOCATION"];
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    // `nested_slope_case(true)` as CSV, labels zero-padded so the factor order
    // is the id order.
    let (x, y, _, n, h, g) = crate::fit::common_tests::nested_slope_case(true);
    let mut nested_csv = String::from("y,x,h,g\n");
    for i in 0..n {
        nested_csv += &format!("{},{},h{:02},g{:02}\n", y[i], x[2 * i + 1], h[i], g[i]);
    }
    let shapes: [(&str, &[&str], &str, Family); 11] = [
        (
            include_str!("../../validation/data/empirical/Penicillin.csv"),
            &["plate", "sample"],
            "diameter ~ 1 + (1 | plate) + (1 | sample)",
            Family::Gaussian,
        ),
        (
            include_str!("../../validation/data/empirical/Pastes.csv"),
            &["batch", "cask", "sample"],
            "strength ~ 1 + (1 | batch/cask)",
            Family::Gaussian,
        ),
        (
            include_str!("../../validation/data/simulated/sim_nested_crossed_mix.csv"),
            mix_f,
            "y ~ 1 + x + (1 | g1/g2) + (1 | c1)",
            Family::Gaussian,
        ),
        (
            SLEEP,
            &["Subject"],
            "Reaction ~ Days + (1 + Days | Subject)",
            Family::Gaussian,
        ),
        (
            SLEEP,
            &["Subject"],
            "Reaction ~ Days + (Days || Subject)",
            Family::Gaussian,
        ),
        (
            SLEEP,
            &["Subject"],
            "Reaction ~ 1 + (Days | Subject)",
            Family::Gaussian,
        ),
        (
            include_str!("../../validation/data/simulated/sim_cloglog_nested_crossed.csv"),
            mix_f,
            "y ~ 1 + x + (1 | g1/g2) + (1 | c1)",
            Family::Binomial {
                link: BinomialLink::Cloglog,
            },
        ),
        (
            grouse,
            grouse_f,
            "TICKS ~ YEAR + cHEIGHT + (1 | LOCATION) + (1 | BROOD)",
            poisson,
        ),
        (
            grouse,
            grouse_f,
            "TICKS ~ YEAR + cHEIGHT + (1 | BROOD) + (1 | LOCATION)",
            poisson,
        ),
        (
            grouse,
            grouse_f,
            "TICKS ~ YEAR + cHEIGHT + (1 | LOCATION) + (1 | BROOD) + (1 | INDEX)",
            poisson,
        ),
        (
            &nested_csv,
            &["h", "g"],
            "y ~ 1 + x + (1 | h) + (1 + x | g)",
            Family::Gaussian,
        ),
    ];
    let opts = PredictOptions {
        se: true,
        ..PredictOptions::default()
    };
    let mut ran = 0;
    for (csv, factors, formula, family) in shapes {
        let (lo, fit) = lowered_fit(csv, factors, formula, family);
        if csv == nested_csv {
            let c = fit.condvar_cross.as_ref().unwrap();
            assert!(!c.nested_slots.is_empty(), "{formula}: the nested map");
        }
        let joint = crate::fit::common_tests::dense_joint(&lo, &fit);
        let mut gap = 0.0f64;
        let (n, p) = (lo.n, lo.p);
        let kt = joint.cv.nrows();
        let starts: Vec<usize> = joint
            .groups
            .iter()
            .scan(0, |at, &(lv, q)| {
                let s = *at;
                *at += lv * q;
                Some(s)
            })
            .collect();
        // Rows 3i+1 lose their grouping-0 level, rows 3i+2 their last
        // grouping's (on a single grouping, grouping 0 again); rows 3i move a
        // nested level to the same child slot of the next parent.
        let last = joint.groups.len() - 1;
        let nested = lo
            .spec()
            .re
            .as_ref()
            .unwrap()
            .extra_groupings
            .iter()
            .position(|g| g.relation == crate::GroupingRelation::NestedWithin);
        let mut ids = lo.ids.clone();
        for i in 0..n {
            match i % 3 {
                1 => ids.primary[i] = joint.groups[0].0 as u32,
                2 if last == 0 => ids.primary[i] = joint.groups[0].0 as u32,
                2 => ids.extra[last - 1][i] = joint.groups[last].0 as u32,
                _ => {
                    if let Some(e) = nested {
                        let slots = joint.groups[e + 1].0;
                        let m = slots / joint.groups[0].0;
                        ids.extra[e][i] = ((ids.extra[e][i] as usize + m) % slots) as u32;
                    }
                }
            }
        }
        let id_of = |ids: &GroupIds, g: usize, i: usize| {
            if g == 0 {
                ids.primary[i]
            } else {
                ids.extra[g - 1][i]
            }
        };
        let design = lo.data().design;
        for (changed, new) in [(false, design), (true, design.ids(&ids))] {
            let pred = predict(&fit, lo.spec(), &new, &opts).unwrap();
            for i in 0..n {
                let mut z: Vec<f64> = (0..kt).map(|k| joint.z[(i, k)]).collect();
                // A changed level clears its grouping's columns; a seen one is
                // an intercept-only indicator.
                for (g, &(levels, q)) in joint.groups.iter().enumerate() {
                    let id = id_of(&ids, g, i) as usize;
                    if !changed || id == id_of(&lo.ids, g, i) as usize {
                        continue;
                    }
                    z[starts[g]..starts[g] + levels * q].fill(0.0);
                    if id < levels {
                        assert_eq!(q, 1);
                        z[starts[g] + id] = 1.0;
                    }
                }
                let mut var: f64 = (0..kt)
                    .flat_map(|a| (0..kt).map(move |b| (a, b)))
                    .map(|(a, b)| z[a] * joint.cv[(a, b)] * z[b])
                    .sum();
                let d: Vec<f64> = (0..p)
                    .map(|j| {
                        lo.x[i * p + j] - (0..kt).map(|k| joint.kx[(k, j)] * z[k]).sum::<f64>()
                    })
                    .collect();
                for j in 0..p {
                    for k in 0..p {
                        var += d[j] * fit.vcov()[j * p + k] * d[k];
                    }
                }
                let want = var.sqrt();
                assert!(
                    (pred.se[i] - want).abs() <= ROUNDOFF * (1.0 + want),
                    "{formula}, changed ids {changed}, row {i}: {} vs {want}",
                    pred.se[i]
                );
                gap = gap.max((pred.se[i] - want).abs() / (1.0 + want));
            }
        }
        eprintln!("{formula}: conditional SE {gap:e}");
        ran += 1;
    }
    assert_eq!(ran, 11);
}

/// The inputs `predict` refuses, one per check.
#[test]
fn option_errors() {
    let (sleep, sleep_fit) = lowered_fit(
        SLEEP,
        &["Subject"],
        "Reaction ~ Days + (1 + Days | Subject)",
        Family::Gaussian,
    );
    let (glm, glm_fit) = lowered_fit(BERN, &["subj", "g", "h"], "y ~ x + g", binomial());
    let (sparse, sparse_fit) = lowered_fit(
        include_str!("../../validation/data/simulated/sim_slope_extra.csv"),
        &["g1", "g2"],
        "y ~ 1 + x + (1 + x | g1) + (1 + x | g2)",
        Family::Gaussian,
    );
    let wide: Vec<f64> = sleep.x.chunks(2).flat_map(|r| [r[0], r[1], 0.0]).collect();
    let d = PredictOptions::default;
    type Case = (
        &'static str,
        Result<Prediction, FitError>,
        fn(&FitError) -> bool,
    );
    let cases: Vec<Case> = vec![
        (
            "level 0",
            predict(
                &sleep_fit,
                sleep.spec(),
                &sleep.data().design,
                &PredictOptions { level: 0.0, ..d() },
            ),
            |e| matches!(e, FitError::OptionValue(_)),
        ),
        (
            "level 1",
            predict(
                &sleep_fit,
                sleep.spec(),
                &sleep.data().design,
                &PredictOptions { level: 1.0, ..d() },
            ),
            |e| matches!(e, FitError::OptionValue(_)),
        ),
        (
            "future on an LMM",
            predict(
                &sleep_fit,
                sleep.spec(),
                &sleep.data().design,
                &PredictOptions {
                    future: true,
                    ..d()
                },
            ),
            |e| matches!(e, FitError::OptionValue(_)),
        ),
        (
            "future on a GLM",
            predict(
                &glm_fit,
                glm.spec(),
                &glm.data().design,
                &PredictOptions {
                    future: true,
                    ..d()
                },
            ),
            |e| matches!(e, FitError::OptionValue(_)),
        ),
        (
            "a design with another p",
            predict(
                &sleep_fit,
                sleep.spec(),
                &Design::new(&wide, sleep.n, 3).ids(&sleep.ids),
                &d(),
            ),
            |e| matches!(e, FitError::Shape(_)),
        ),
        (
            "conditional without ids",
            predict(
                &sleep_fit,
                sleep.spec(),
                &Design::new(&sleep.x, sleep.n, sleep.p),
                &d(),
            ),
            |e| matches!(e, FitError::Ids(_)),
        ),
        (
            "conditional SE on the sparse route",
            predict(
                &sparse_fit,
                sparse.spec(),
                &sparse.data().design,
                &PredictOptions { se: true, ..d() },
            ),
            |e| matches!(e, FitError::Unsupported(_)),
        ),
    ];
    assert!(
        sparse_fit.condvar.is_empty()
            && !sparse_fit.ranef.is_empty()
            && sparse_fit.ranef.iter().all(|v| v.is_finite())
    );
    let mut ran = 0;
    for (what, got, want) in &cases {
        match got {
            Err(e) => assert!(want(e), "{what}: {e:?}"),
            Ok(_) => panic!("{what}: accepted"),
        }
        ran += 1;
    }
    assert_eq!(ran, 7);
    // Population mode needs no ids and no conditional variances.
    let pop = PredictOptions {
        population: true,
        se: true,
        ..d()
    };
    assert!(predict(
        &sleep_fit,
        sleep.spec(),
        &Design::new(&sleep.x, sleep.n, 2),
        &pop
    )
    .is_ok());
    assert!(predict(&sparse_fit, sparse.spec(), &sparse.data().design, &pop).is_ok());
}

/// A new row past the data range is refused like a fit's row, since `predict`
/// runs the fit's design check.
#[test]
fn a_prediction_past_the_data_range_is_refused() {
    let spec = ModelSpec::fixed(Family::Gaussian);
    let (x, y) = ([1.0; 4], [1.0, 2.0, 3.0, 4.0]);
    let fit = crate::fit(&Data::new(&x, &y, 4, 1), &spec, &FitOptions::default())
        .unwrap_or_else(|e| panic!("{e:?}"));
    let new = [1.0, 2.0 * crate::limits::MAX_DATA_ABS];
    let got = predict(
        &fit,
        &spec,
        &Design::new(&new, 2, 1),
        &PredictOptions::default(),
    );
    match got {
        Err(FitError::Shape(m)) => assert!(m.contains("glmm::limits::MAX_DATA_ABS"), "{m:?}"),
        Err(e) => panic!("wrong refusal: {e:?}"),
        Ok(_) => panic!("accepted"),
    }
}

/// The Monte Carlo check of the future-observation bounds.
#[cfg(feature = "montecarlo-tests")]
mod montecarlo {
    use super::*;
    use crate::NegBinomialLink;
    use commonstats::dist::{Bernoulli, DiscreteSampler, Normal, Sampler};
    use commonstats::rng::CommonStatsRng;

    /// The stream of the mixture draws, distinct from every `commonstats`
    /// `STREAM_TAG_*` and from the crate's other Monte Carlo tags.
    const PREDMIX_TAG: u64 = u64::from_be_bytes(*b"PREDMIX1");
    /// Draws of the mixture per row.
    const DRAWS: u64 = 200_000;
    /// Binomial standard errors an empirical CDF may sit from its
    /// probability: the Monte Carlo band, as `rlrt_refs::bands::RLRT_LEVEL_SE`.
    /// Set a priori.
    const MC_SE: f64 = 3.0;

    /// On the Bernoulli, Poisson, Gamma and negative-binomial GLMMs, three
    /// rows each (the first, middle and last training rows taken as new
    /// rows): `DRAWS` draws of `η ~ N(η̂, se²)` (the row's conditional link
    /// fit and standard error) and then of the family at `μ = g⁻¹(η)` with
    /// the fitted dispersion. At `p = (1 + level)/2` the empirical CDF at
    /// [`Prediction::future_upper`] reaches `p`, and on a discrete family the
    /// CDF one count below it stays under `p`, each within [`MC_SE`] binomial
    /// standard errors; on the Gamma the CDF there equals `p` within them.
    #[test]
    fn future_quantiles_match_a_monte_carlo_of_the_mixture() {
        let gamma = Family::Gamma {
            link: GammaLink::Log,
        };
        let nb = Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        };
        let shapes: [(&str, &[&str], &str, Family); 4] = [
            (
                BERN,
                &["subj", "g", "h"],
                "y ~ g * h + x + (1 | subj)",
                binomial(),
            ),
            (
                include_str!("../../validation/data/simulated/sim_poisson_offset.csv"),
                &["cluster"],
                "y ~ 1 + x + offset(log_exposure) + (1 | cluster)",
                Family::Poisson {
                    link: PoissonLink::Log,
                },
            ),
            (
                include_str!("../../validation/data/simulated/sim_gamma.csv"),
                &["cluster", "grp"],
                "y ~ 1 + x + grp + (1 | cluster)",
                gamma,
            ),
            (
                include_str!("../../validation/data/simulated/sim_nb.csv"),
                &["cluster", "grp"],
                "y ~ 1 + x + grp + (1 | cluster)",
                nb,
            ),
        ];
        let normal = Normal::new(0.0, 1.0).unwrap();
        let opts = PredictOptions {
            se: true,
            future: true,
            ..PredictOptions::default()
        };
        let p = 0.5 + opts.level / 2.0;
        let band = MC_SE * (p * (1.0 - p) / DRAWS as f64).sqrt();
        let mut checked = 0;
        for (m, (csv, factors, formula, family)) in shapes.into_iter().enumerate() {
            let (lo, fit) = lowered_fit(csv, factors, formula, family);
            assert!(matches!(lo.spec().re, Some(ref re) if *re == Re::intercept()));
            let (n, k) = (lo.n, lo.p);
            let rows = [0, n / 2, n - 1];
            let x: Vec<f64> = rows
                .iter()
                .flat_map(|&i| lo.x[i * k..(i + 1) * k].to_vec())
                .collect();
            let ids = GroupIds::new(rows.iter().map(|&i| lo.ids.primary[i]).collect());
            let offset: Option<Vec<f64>> = lo
                .offset
                .as_ref()
                .map(|o| rows.iter().map(|&i| o[i]).collect());
            let mut new = Design::new(&x, 3, k).ids(&ids);
            if let Some(o) = &offset {
                new = new.offset(o);
            }
            let pred = predict(&fit, lo.spec(), &new, &opts).unwrap();
            let phi = fit.dispersion;
            for (r, &row) in rows.iter().enumerate() {
                let upper = pred.future_upper[r];
                assert!(upper.is_finite(), "{formula} row {r}");
                let mut rng = CommonStatsRng::new_tagged(35, (3 * m + r) as u64, PREDMIX_TAG);
                let (mut at, mut below) = (0u64, 0u64);
                for _ in 0..DRAWS {
                    let eta = pred.fit[r] + pred.se[r] * normal.sample(&mut rng);
                    let mu = link_inv(family, eta);
                    let y = match family {
                        Family::Binomial { .. } => {
                            Bernoulli::new(mu).unwrap().sample(&mut rng) as f64
                        }
                        Family::Poisson { .. } => Poisson::new(mu).unwrap().sample(&mut rng) as f64,
                        Family::NegativeBinomial { .. } => NegBinomial::from_mean_size(mu, phi)
                            .unwrap()
                            .sample(&mut rng)
                            as f64,
                        _ => Gamma::new(1.0 / phi, 1.0 / (phi * mu))
                            .unwrap()
                            .sample(&mut rng),
                    };
                    at += u64::from(y <= upper);
                    below += u64::from(y <= upper - 1.0);
                }
                let f_at = at as f64 / DRAWS as f64;
                let f_below = below as f64 / DRAWS as f64;
                println!(
                    "{formula}, {family:?}, row {}: upper {upper}, CDF there {f_at:.5}, one below {f_below:.5}",
                    row
                );
                if family == gamma {
                    assert!((f_at - p).abs() <= band, "{formula} row {r}: {f_at}");
                } else {
                    assert!(f_at >= p - band, "{formula} row {r}: CDF {f_at} at {upper}");
                    assert!(
                        upper < 1.0 || f_below <= p + band,
                        "{formula} row {r}: CDF {f_below} below {upper}"
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 12);
    }
}
