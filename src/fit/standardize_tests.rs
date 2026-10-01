//! Internal standardization of the design, end to end: unit invariance on
//! every route (one table), the intercept found wherever it sits, a reused
//! workspace, a carried θ̂, and a pinned block with a centered slope.

use super::common::spec_sized_from_ids_pub;
use super::common_tests::{lcg, sim_clustered};
use super::core::{build_workspace, fit_on};
use super::*;
use crate::{ReStructure, Sizing};

/// One route under test: design, response, spec, ids, options, and which
/// column the arms act on.
#[derive(Clone)]
struct Case {
    name: &'static str,
    x: Vec<f64>,
    y: Vec<f64>,
    n: usize,
    p: usize,
    model: ModelSpec,
    ids: GroupIds,
    opts: FitOptions,
    col: usize,
}

/// `cluster,x,grp,y` fixture with the given family and random structure.
fn clustered(
    name: &'static str,
    csv: &str,
    family: Family,
    slopes: Vec<u32>,
    re: bool,
    nagq: u8,
) -> Case {
    let (x, y, cids, nc) = sim_clustered(csv);
    let n = y.len();
    Case {
        name,
        x,
        y,
        n,
        p: 3,
        model: ModelSpec {
            family,
            re: re.then(|| ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: nc as u32,
                },
                slopes,
                extra_groupings: vec![],
            }),
        },
        ids: if re {
            GroupIds {
                primary: cids,
                extra: vec![],
            }
        } else {
            GroupIds::default()
        },
        opts: FitOptions {
            target_indices: vec![0, 1, 2],
            nagq,
            ..FitOptions::default()
        },
        col: 1,
    }
}

/// `y,x,g` fixtures (sim_binomial_slope1, sim_poisson_slope1): X = [1, x].
fn yxg(name: &'static str, csv: &str, family: Family, slopes: Vec<u32>, nagq: u8) -> Case {
    let mut x = vec![];
    let mut y = vec![];
    let mut raw = vec![];
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse::<f64>().unwrap());
        x.extend_from_slice(&[1.0, f[1].parse::<f64>().unwrap()]);
        raw.push(f[2].to_string());
    }
    let mut levels: Vec<String> = raw.clone();
    levels.sort();
    levels.dedup();
    let ids: Vec<u32> = raw
        .iter()
        .map(|r| levels.binary_search(r).unwrap() as u32)
        .collect();
    let n = y.len();
    Case {
        name,
        x,
        y,
        n,
        p: 2,
        model: ModelSpec {
            family,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: levels.len() as u32,
                },
                slopes,
                extra_groupings: vec![],
            }),
        },
        ids: GroupIds {
            primary: ids,
            extra: vec![],
        },
        opts: FitOptions {
            target_indices: vec![0, 1],
            nagq,
            ..FitOptions::default()
        },
        col: 1,
    }
}

fn cases() -> Vec<Case> {
    let gamma = include_str!("../../validation/data/simulated/sim_gamma.csv");
    let nb = include_str!("../../validation/data/simulated/sim_nb.csv");
    let bin = include_str!("../../validation/data/simulated/sim_binomial_slope1.csv");
    let logit = Family::Binomial {
        link: crate::BinomialLink::Logit,
    };
    let nbf = Family::NegativeBinomial {
        link: crate::NegBinomialLink::Log,
    };
    let mut v = vec![
        clustered("ols", gamma, Family::Gaussian, vec![], false, 1),
        clustered(
            "glm",
            gamma,
            Family::Gamma {
                link: crate::GammaLink::Log,
            },
            vec![],
            false,
            1,
        ),
        clustered("lmm_dense", gamma, Family::Gaussian, vec![1], true, 1),
        yxg("glmm_laplace", bin, logit, vec![1], 1),
        yxg("glmm_scalar_agq", bin, logit, vec![], 5),
        yxg("glmm_vector_agq", bin, logit, vec![1], 5),
        clustered("nb_glm", nb, nbf, vec![], false, 1),
        clustered("nb_glmm", nb, nbf, vec![], true, 1),
    ];
    let (x, y, n, p, model, ids) = crate::sparse::tests::sparse_gamma_design();
    v.push(Case {
        name: "lmm_sparse",
        x,
        y,
        n,
        p,
        model: ModelSpec {
            family: Family::Gaussian,
            ..model
        },
        ids,
        opts: FitOptions {
            target_indices: (0..p as u32).collect(),
            ..FitOptions::default()
        },
        col: 1,
    });
    v
}

fn with_column(c: &Case, s: f64, a: f64) -> Vec<f64> {
    let mut x = c.x.clone();
    for row in x.chunks_exact_mut(c.p) {
        row[c.col] = row[c.col] * s + a;
    }
    x
}

fn fit(c: &Case, x: &[f64]) -> Fit {
    fit_cold(x, &c.y, c.n, c.p, &c.model, &c.ids, &c.opts)
}

/// Relative closeness with an absolute floor for values near zero.
fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * (1.0 + b.abs())
}

/// `cases()` plus the routes and inputs only the unit-invariance table needs:
/// Gamma GLMM (Laplace and scalar AGQ), the `Rx` Wald arm, a sparse GLMM on
/// binomial trial weights, and precision weights (with an offset on the GLMM).
fn invariance_cases() -> Vec<Case> {
    let gamma = include_str!("../../validation/data/simulated/sim_gamma.csv");
    let gammaf = Family::Gamma {
        link: crate::GammaLink::Log,
    };
    let mut v = cases();
    v.push(clustered("glmm_gamma", gamma, gammaf, vec![], true, 1));
    v.push(clustered("glmm_gamma_agq", gamma, gammaf, vec![], true, 5));
    let named = |v: &[Case], name: &str| v.iter().find(|c| c.name == name).unwrap().clone();
    let mut rx = named(&v, "glmm_laplace");
    rx.name = "glmm_rx";
    rx.opts.wald_se = WaldSe::Rx;
    v.push(rx);
    let (x, y, n, p, model, ids, w) = crate::sparse::tests::sparse_glmm_slope_crossed_design();
    v.push(Case {
        name: "glmm_sparse",
        x,
        y,
        n,
        p,
        model,
        ids,
        opts: FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w),
            ..FitOptions::default()
        },
        col: 1,
    });
    for (from, name) in [
        ("ols", "ols_weighted"),
        ("lmm_dense", "lmm_weighted"),
        ("glmm_laplace", "glmm_weighted_offset"),
    ] {
        let mut c = named(&v, from);
        c.name = name;
        let mut st = 7u64;
        // `lcg` is U(−1, 1): weights in (0.5, 1.5).
        c.opts.weights = Some((0..c.n).map(|_| 1.0 + 0.5 * lcg(&mut st)).collect());
        if c.model.re.is_some() && !matches!(c.model.family, Family::Gaussian) {
            c.opts.offset = Some((0..c.n).map(|_| 0.2 * lcg(&mut st)).collect());
        }
        v.push(c);
    }
    v
}

/// Bands per route class on the parameters: β, vcov (relative to
/// √(V_aa·V_bb)), varcorr, tau2, the criteria, dispersion and stddev_se.
/// Measured 2026-10-01 on dell-fedora, debug and release alike, the worst
/// relative deviation over the table's arms: OLS and GLM 2.8e-15 (β); the NB,
/// LMM and GLMM routes at most 3.7e-8 (tau2 of the weighted GLMM with an
/// offset); the two sparse routes 5.0e-5 (tau2). Each band is the worst
/// ×10, rounded up to a power of ten, except the sparse one, held at the 1e-4
/// it had (their wide θ blocks stop within BOBYQA's tolerance on a different
/// path for each arm).
const BAND_CLOSED: f64 = 1e-13; // OLS, GLM: no outer search
const BAND_SEARCH: f64 = 1e-6; // LMM, GLMM, NB
const BAND_SPARSE: f64 = 1e-4; // the sparse routes' wide θ blocks

/// `(parameters, modes)` bands of a route. The conditional modes and the
/// fitted values move further than the parameters: the modes inherit the
/// search's stopping point amplified, and a fitted value on the shifted arm
/// carries the round-off of `x + 1000` itself. Measured as above: 2.6e-13
/// (OLS fitted), 4.3e-7 (ranef of the weighted GLMM with an offset), 5.3e-5
/// (sparse LMM fitted). Same rule, with the sparse band held at 3e-4.
fn band(c: &Case) -> (f64, f64) {
    if c.name.ends_with("sparse") {
        (BAND_SPARSE, 3e-4)
    } else if c.model.re.is_none() && !matches!(c.model.family, Family::NegativeBinomial { .. }) {
        (BAND_CLOSED, 1e-12)
    } else {
        (BAND_SEARCH, 1e-5)
    }
}

/// The arm `x → s·x + a` on the acted-on column fits the same model, with
/// every coefficient vector `v` whose entry 0 is the intercept and entry `k`
/// the column's coefficient read as `T·v`: `v₀ − (a/s)·v_k` and `v_k/s`.
fn reparam(v: &mut [f64], k: usize, s: f64, a: f64) {
    v[0] -= a / s * v[k];
    v[k] /= s;
}

/// `T·M·Tᵀ` of a symmetric matrix given by rows.
fn congruent(m: &[Vec<f64>], k: usize, s: f64, a: f64) -> Vec<Vec<f64>> {
    let mut m = m.to_vec();
    for row in m.iter_mut() {
        reparam(row, k, s, a);
    }
    for c in 0..m.len() {
        let mut col: Vec<f64> = m.iter().map(|r| r[c]).collect();
        reparam(&mut col, k, s, a);
        for (r, v) in m.iter_mut().zip(col) {
            r[c] = v;
        }
    }
    m
}

/// Column-major lower vech (the `varcorr` layout) to the full symmetric block.
fn unvech(v: &[f64]) -> Vec<Vec<f64>> {
    let q = (1..=v.len()).find(|q| q * (q + 1) / 2 == v.len()).unwrap();
    let mut m = vec![vec![0.0; q]; q];
    let slots = (0..q).flat_map(|c| (c..q).map(move |r| (r, c)));
    for ((r, c), &x) in slots.zip(v) {
        m[r][c] = x;
        m[c][r] = x;
    }
    m
}

/// The lower triangle of `m` in vech order.
fn vech(m: &[Vec<f64>]) -> Vec<f64> {
    let q = m.len();
    (0..q).flat_map(|c| (c..q).map(move |r| m[r][c])).collect()
}

/// The entrywise squares of the lower Cholesky factor of `m`, in vech order:
/// what `tau2` reports for a block whose covariance is `m`. A pivot that
/// rounds below zero is a zero (a rank-deficient block), and a zero pivot's
/// column is zero below it, as in the canonical factor the fit reports.
fn chol_squared(m: &[Vec<f64>]) -> Vec<f64> {
    let q = m.len();
    let mut l = vec![vec![0.0; q]; q];
    for c in 0..q {
        let pivot = m[c][c] - (0..c).map(|k| l[c][k] * l[c][k]).sum::<f64>();
        l[c][c] = pivot.max(0.0).sqrt();
        for r in c + 1..q {
            let below = m[r][c] - (0..c).map(|k| l[r][k] * l[c][k]).sum::<f64>();
            l[r][c] = if l[c][c] > 0.0 { below / l[c][c] } else { 0.0 };
        }
    }
    vech(&l).iter().map(|v| v * v).collect()
}

/// Per grouping, in declaration order: the block width and the acted-on
/// column's row in the block (`None` when the block does not read it).
fn blocks(c: &Case) -> Vec<(usize, Option<usize>)> {
    let Some(re) = &c.model.re else {
        return vec![];
    };
    std::iter::once(&re.slopes)
        .chain(re.extra_groupings.iter().map(|g| &g.slopes))
        .map(|sl| {
            let row = sl.iter().position(|&s| s as usize == c.col).map(|i| i + 1);
            (1 + sl.len(), row)
        })
        .collect()
}

/// Rescaling or shifting a column refits the same model on every route: each
/// caller-unit result moves exactly as the reparameterization `T` predicts.
/// β → T·β and vcov → T·V·Tᵀ, se its diagonal (under a shift the intercept's
/// reads the full V). Each random-effect block that reads the column moves as
/// D → T·D·Tᵀ, its tau2 the squared Cholesky factor of that, and each level's
/// modes as b → T·b. On a rescale the GLMM stddev SEs of the column's rows
/// divide by s; a shift moves the intercept's stddev nonlinearly, so its
/// block's SEs are not predicted there. Fitted values, dispersion and the ML
/// criteria do not move; through log|X'V⁻¹X| the REML deviance moves by
/// 2·ln s and the REML loglik by −ln s.
#[test]
fn unit_invariance_on_every_route() {
    for c in invariance_cases() {
        let (band, modes) = band(&c);
        let base = fit(&c, &c.x);
        assert!(base.converged(), "{}: base", c.name);
        let blocks = blocks(&c);
        assert_eq!(base.varcorr.len(), blocks.len(), "{}: blocks", c.name);
        let mut worst = (0.0f64, String::new());
        for (s, a) in [(1e3, 0.0), (1e-3, 0.0), (1.0, 1000.0)] {
            let f = fit(&c, &with_column(&c, s, a));
            let arm = format!("{} x·{s:e}+{a}", c.name);
            assert!(f.converged(), "{arm}");
            let mut check = |what: &str, got: &[f64], want: &[f64]| {
                assert_eq!(got.len(), want.len(), "{arm}: {what} length");
                for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
                    if g.is_nan() && w.is_nan() {
                        continue;
                    }
                    let rel = (g - w).abs() / (1.0 + w.abs());
                    if rel > worst.0 {
                        worst = (rel, format!("{arm}: {what}[{i}]"));
                    }
                    let b = if matches!(what, "ranef" | "fitted") {
                        modes
                    } else {
                        band
                    };
                    assert!(close(g, w, b), "{arm}: {what}[{i}] {g} vs {w}");
                }
            };
            let mut beta = base.beta.clone();
            reparam(&mut beta, c.col, s, a);
            check("β", &f.beta, &beta);
            let v = congruent(&base.vcov, c.col, s, a);
            let se: Vec<f64> = (0..c.p).map(|k| v[k][k].sqrt()).collect();
            check("se", &f.se, &se);
            // vcov relative to √(V_aa·V_bb): an off-diagonal can be near zero.
            let vcov_rel: Vec<f64> = (0..c.p * c.p)
                .map(|i| {
                    (f.vcov[i / c.p][i % c.p] - v[i / c.p][i % c.p]) / (se[i / c.p] * se[i % c.p])
                })
                .collect();
            check("vcov", &vcov_rel, &vec![0.0; c.p * c.p]);
            let reml = if f.reml { s.ln() } else { 0.0 };
            if base.deviance.is_finite() {
                check("deviance", &[f.deviance], &[base.deviance + 2.0 * reml]);
            }
            check("loglik", &[f.loglik], &[base.loglik - reml]);
            check("dispersion", &[f.dispersion], &[base.dispersion]);
            check("fitted", &f.fitted, &base.fitted);
            let (mut t, mut r) = (0, 0);
            let (mut tau2, mut ranef, mut sd_se) = (vec![], vec![], vec![]);
            for (g, &(q, row)) in blocks.iter().enumerate() {
                let d = unvech(&base.varcorr[g]);
                let d = match row {
                    Some(k) => congruent(&d, k, s, a),
                    None => d,
                };
                check(&format!("varcorr[{g}]"), &f.varcorr[g], &vech(&d));
                tau2.extend(chol_squared(&d));
                let levels = base.ranef_levels[g];
                for b in base.ranef[r..r + levels * q].chunks_exact(q) {
                    let mut b = b.to_vec();
                    if let Some(k) = row {
                        reparam(&mut b, k, s, a);
                    }
                    ranef.extend(b);
                }
                r += levels * q;
                let len = q * (q + 1) / 2;
                // Empty on the LMM routes.
                if let Some(se) = base.stddev_se.get(t..t + len) {
                    let mut se = se.to_vec();
                    match row {
                        // Row k's diagonal slot: column k opens at k·q − k(k−1)/2.
                        Some(k) if a == 0.0 => se[k * q - k * (k - 1) / 2] /= s,
                        // Not predicted under a shift: the fit's own values.
                        Some(_) => se.copy_from_slice(&f.stddev_se[t..t + len]),
                        None => {}
                    }
                    sd_se.extend(se);
                }
                t += len;
            }
            check("tau2", &f.tau2, &tau2);
            check("ranef", &f.ranef, &ranef);
            check("stddev_se", &f.stddev_se, &sd_se);
        }
        eprintln!("{}: worst {:.1e} at {}", c.name, worst.0, worst.1);
    }
}

/// The intercept found wherever it sits: moving it from column 0 to the last
/// column changes no caller-unit result.
#[test]
fn intercept_position_does_not_change_the_fit() {
    for c in cases().into_iter().filter(|c| {
        c.model.re.is_none() || matches!(c.name, "glmm_laplace" | "lmm_dense" | "lmm_sparse")
    }) {
        let band = band(&c).0.max(BAND_SEARCH);
        // Move column 0 (the intercept) to the last position: new column
        // `new` holds old column `order[new]`.
        let p = c.p;
        let order: Vec<usize> = (1..p).chain(std::iter::once(0)).collect();
        let new_of = |old: usize| order.iter().position(|&o| o == old).unwrap();
        let mut c2 = c.clone();
        c2.x =
            c.x.chunks_exact(p)
                .flat_map(|r| order.iter().map(move |&j| r[j]))
                .collect();
        c2.col = new_of(c.col);
        if let Some(re) = c2.model.re.as_mut() {
            let remap = |sl: &mut Vec<u32>| {
                for s in sl.iter_mut() {
                    *s = new_of(*s as usize) as u32;
                }
            };
            remap(&mut re.slopes);
            for g in &mut re.extra_groupings {
                remap(&mut g.slopes);
            }
        }
        let (a, b) = (fit(&c, &c.x), fit(&c2, &c2.x));
        for (new, &old) in order.iter().enumerate() {
            assert!(
                close(b.beta[new], a.beta[old], band),
                "{}: β[{old}]",
                c.name
            );
            assert!(close(b.se[new], a.se[old], band), "{}: se[{old}]", c.name);
        }
    }
}

/// A reused loop-tier workspace recomputes the statistics per draw, including
/// a draw where the acted-on column turns 0/1: each draw equals `fit_cold`.
#[test]
fn fit_on_reuse_recomputes_statistics_per_draw() {
    for c in cases()
        .into_iter()
        .filter(|c| matches!(c.name, "ols" | "lmm_dense" | "glmm_laplace"))
    {
        let (sized, ids, perm) = spec_sized_from_ids_pub(&c.model, &c.ids);
        let mut ws = build_workspace(&sized, perm, c.n, c.p, &c.opts);
        let draws = [
            with_column(&c, 1e3, 0.0),
            with_column(&c, 1.0, 1000.0),
            c.x.clone(),
            {
                // the acted-on column turned 0/1
                let mut x = c.x.clone();
                for (i, row) in x.chunks_exact_mut(c.p).enumerate() {
                    row[c.col] = (i % 2) as f64;
                }
                x
            },
        ];
        let mut joint = vec![];
        for (d, x) in draws.iter().enumerate() {
            let view = fit_on(&mut ws, x, &c.y, &ids, None, &c.opts);
            joint.push(view.joint_t_sq());
            let got = view.into_fit(x, &c.y, &ids, c.n, c.p, &c.model, &c.opts);
            let want = fit(&c, x);
            for k in 0..c.p {
                assert!(
                    close(got.beta[k], want.beta[k], 1e-12),
                    "{} draw {d}: β[{k}]",
                    c.name
                );
            }
        }
        // The joint Wald statistic over every coefficient is invariant under
        // the reparameterizations of draws 0 and 1 (NaN on OLS, which has none).
        for d in 0..2 {
            assert!(
                (joint[d].is_nan() && joint[2].is_nan()) || close(joint[d], joint[2], BAND_SEARCH),
                "{} draw {d}: joint Wald {} vs {}",
                c.name,
                joint[d],
                joint[2]
            );
        }
    }
}

/// θ̂ carried from draw k (caller units, declaration order)
/// into draw k+1 whose statistics differ; a wrong-width start is dropped,
/// whatever the scaling.
#[test]
fn warm_theta_carry_across_draws_matches_cold() {
    for c in cases()
        .into_iter()
        .filter(|c| matches!(c.name, "lmm_dense" | "glmm_laplace"))
    {
        let (sized, ids, perm) = spec_sized_from_ids_pub(&c.model, &c.ids);
        let mut ws = build_workspace(&sized, perm, c.n, c.p, &c.opts);
        let x1 = with_column(&c, 1.0, 1000.0);
        let theta = fit_on(&mut ws, &c.x, &c.y, &ids, None, &c.opts)
            .theta()
            .to_vec();
        let start = StartValues {
            beta: vec![],
            theta,
        };
        let warm = fit_on(&mut ws, &x1, &c.y, &ids, Some(&start), &c.opts)
            .into_fit(&x1, &c.y, &ids, c.n, c.p, &c.model, &c.opts);
        let cold = fit(&c, &x1);
        assert!(
            close(warm.deviance, cold.deviance, 1e-7),
            "{}: warm {} vs cold {}",
            c.name,
            warm.deviance,
            cold.deviance
        );
        // θ̂ of the draw x + 1000 carried into the same draw: mapped into the
        // standardized design it is the optimum the search starts at, where
        // read in the wrong units (the intercept variance at x = 0 taken as the
        // one at the mean) it is far from it.
        let theta1 = fit_on(&mut ws, &x1, &c.y, &ids, None, &c.opts)
            .theta()
            .to_vec();
        let again = fit_on(
            &mut ws,
            &x1,
            &c.y,
            &ids,
            Some(&StartValues {
                beta: vec![],
                theta: theta1,
            }),
            &c.opts,
        )
        .into_fit(&x1, &c.y, &ids, c.n, c.p, &c.model, &c.opts);
        assert!(
            close(again.deviance, cold.deviance, 1e-7),
            "{}: carried into the same draw",
            c.name
        );
        // On the GLMM route the search's length follows its start (measured
        // 2026-10-01: 22 evaluations from the carry, 39 cold). The LMM's does
        // not (48 against 47): BOBYQA's trust-region schedule fixes its cost
        // there, so its map is checked by the θ round trip in `lmm::re_map`.
        if c.name == "glmm_laplace" {
            assert!(
                3 * again.n_eval < 2 * cold.n_eval,
                "{}: the carried θ̂ took {} evaluations, the cold fit {}",
                c.name,
                again.n_eval,
                cold.n_eval
            );
        }
        // A θ start of the wrong width is dropped to a cold θ start: the fit is
        // the one a fresh workspace makes with no start at all, bit for bit.
        let short = StartValues {
            beta: vec![],
            theta: vec![0.5],
        };
        let s = fit_on(&mut ws, &x1, &c.y, &ids, Some(&short), &c.opts)
            .into_fit(&x1, &c.y, &ids, c.n, c.p, &c.model, &c.opts);
        let mut fresh = build_workspace(&sized, perm, c.n, c.p, &c.opts);
        let f0 = fit_on(&mut fresh, &x1, &c.y, &ids, None, &c.opts)
            .into_fit(&x1, &c.y, &ids, c.n, c.p, &c.model, &c.opts);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&s.beta), bits(&f0.beta), "{}: short θ start β", c.name);
        assert_eq!(
            s.deviance.to_bits(),
            f0.deviance.to_bits(),
            "{}: short θ start deviance",
            c.name
        );
        assert_eq!(s.n_eval, f0.n_eval, "{}: short θ start evaluations", c.name);
        // The rule does not depend on the scaling: on a workspace that fits the
        // design as given, a β and a θ start of the wrong width are dropped
        // too, with the previous fit's θ̂ still in the workspace.
        ws.set_prestandardized(true);
        fresh.set_prestandardized(true);
        let short = StartValues {
            beta: vec![0.1],
            theta: vec![0.5],
        };
        let s = fit_on(&mut ws, &x1, &c.y, &ids, Some(&short), &c.opts)
            .into_fit(&x1, &c.y, &ids, c.n, c.p, &c.model, &c.opts);
        let f0 = fit_on(&mut fresh, &x1, &c.y, &ids, None, &c.opts)
            .into_fit(&x1, &c.y, &ids, c.n, c.p, &c.model, &c.opts);
        assert_eq!(bits(&s.beta), bits(&f0.beta), "{}: as given, β", c.name);
        assert_eq!(
            s.deviance.to_bits(),
            f0.deviance.to_bits(),
            "{}: as given, deviance",
            c.name
        );
    }
}

/// A pinned block with a centered slope: the pinned-diagonal fixture of
/// `glmm::tests` (the data of
/// `fit_glmm_vector_agq_keeps_search_lambda_at_pinned_diagonal`), x shifted
/// far from zero, through `fit_cold`.
#[test]
fn pinned_fixture_reports_caller_coordinates() {
    let (n, nc) = (400usize, 20usize);
    let (x, y, primary) = crate::glmm::tests::pinned_diagonal_fixture(278, 1000.0);
    let spec = ModelSpec {
        family: Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: nc as u32,
            },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary,
        extra: vec![],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        2,
        &spec,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            nagq: 7,
            ..FitOptions::default()
        },
    );
    assert!(f.converged());
    // Seed 278 lands on the face with the slope diagonal pinned in the
    // caller's coordinates (on the centered design the search pinned the
    // intercept diagonal, so the zero moved with the map).
    let pinned: Vec<bool> = f.diagnostics.pinned.iter().flatten().copied().collect();
    assert_eq!(pinned, vec![false, true], "pinned {pinned:?}");
    assert!(f.diagnostics.singular);
    // The loop tier reports the same exact zero at the slope diagonal of θ̂
    // (column-major vech: entry 2).
    let opts = FitOptions {
        target_indices: vec![0, 1],
        nagq: 7,
        ..FitOptions::default()
    };
    let (sized, ids_s, perm) = spec_sized_from_ids_pub(&spec, &ids);
    let mut ws = build_workspace(&sized, perm, n, 2, &opts);
    let view = fit_on(&mut ws, &x, &y, &ids_s, None, &opts);
    assert_eq!(view.theta()[2], 0.0, "θ̂ {:?}", view.theta());
    assert_eq!(view.diagnostics().pinned_components, 0b10);
    // varcorr is PSD and finite (no false zero from the map).
    let d = &f.varcorr[0];
    assert!(
        d.iter().all(|v| v.is_finite())
            && d[0] >= 0.0
            && d[2] >= 0.0
            && d[1] * d[1] <= d[0] * d[2] * (1.0 + 1e-9)
    );
}

/// The loop tier's GLMM `dispersion()` is D̂₀₀ in the caller's units: the
/// intercept variance `varcorr` reports, not the one at the column mean.
/// `varcorr` sums row 0 of N·Λ̃ squared; `dispersion()` squares Λ₀₀ of the
/// caller's lower factor, √ of that sum, so the two agree to round-off.
#[test]
fn glmm_view_dispersion_is_the_caller_intercept_variance() {
    let c = cases()
        .into_iter()
        .find(|c| c.name == "glmm_laplace")
        .unwrap();
    let x = with_column(&c, 1.0, 1000.0);
    let (sized, ids, perm) = spec_sized_from_ids_pub(&c.model, &c.ids);
    let mut ws = build_workspace(&sized, perm, c.n, c.p, &c.opts);
    let view = fit_on(&mut ws, &x, &c.y, &ids, None, &c.opts);
    let d = view.dispersion();
    let f = view.into_fit(&x, &c.y, &ids, c.n, c.p, &c.model, &c.opts);
    let v = f.varcorr[0][0];
    assert!((d - v).abs() <= 1e-14 * v, "{d} vs {v}");
}
