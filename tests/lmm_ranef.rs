#![cfg(feature = "formula")]
//! LMM conditional modes: the recovery is checked NUMERICALLY against a
//! brute-force solve of the penalized normal equations, and the labelling is
//! checked as NAMES.
//!
//! Two different claims, deliberately tested two different ways.
//!
//! The recovery can be wrong numerically, so it gets an oracle that shares no
//! code with it: [`brute_ranef`] builds `Z` explicitly in `Fit::ranef`'s PUBLIC
//! layout, forms `A = Λ′Z′ZΛ + I` densely, and solves. It knows nothing about
//! the kernel's elimination order, its family blocking, its crossed tail, or its
//! balanced-collapse shortcut — which is the point: every one of those is a
//! route the recovery takes and the oracle does not.
//!
//! The labelling can be wrong SILENTLY — a value-only comparison passes straight
//! through a naming bug — so every labelling assertion here checks names, on a
//! factor whose levels are deliberately not in sorted order.

use glmm::formula::{label_ranef, lower, Column, RanefBlock, Table};
use glmm::{fit_cold, Family, Fit, Note};

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Deterministic pseudo-data in (−1, 1) — the LCG the kernel's own tests use,
/// restated because `src/test_support.rs` is not reachable from an integration
/// test.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn numeric(v: Vec<f64>) -> Column {
    Column::Numeric(v)
}

/// A factor with the caller's OWN level order, not a sorted one — the level
/// order is what treatment contrasts and `ranef` row labels both read.
fn factor(levels: &[&str], labels: &[&str]) -> Column {
    let levels: Vec<String> = strs(levels);
    let codes = labels
        .iter()
        .map(|l| levels.iter().position(|v| v == l).expect("declared level") as u32)
        .collect();
    Column::Factor { levels, codes }
}

/// `y = 0.5 + 0.4·x1 − 0.2·x2 + (per-group shift) + noise`, `n` rows over
/// `groups` cyclically assigned labels.
fn sim(n: usize, seed: u64, group_labels: &[&str]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>) {
    let mut st = seed;
    let g = group_labels.len();
    let shift: Vec<f64> = (0..g).map(|_| 0.6 * lcg(&mut st)).collect();
    let (mut x1, mut x2, mut y, mut lab) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let c = i % g;
        let a = lcg(&mut st);
        let b = lcg(&mut st);
        x1.push(a);
        x2.push(b);
        y.push(0.5 + 0.4 * a - 0.2 * b + shift[c] + 0.8 * lcg(&mut st));
        lab.push(group_labels[c].to_string());
    }
    (x1, x2, y, lab)
}

// ── Brute-force oracle ───────────────────────────────────────────────────────

/// `b̂` by direct dense solve, in `Fit::ranef`'s public layout.
///
/// Solves `(Λ′Z′ZΛ + I) û = Λ′Z′(y − Xβ̂)` and returns `b̂ = Λû`. `Λ` is
/// recovered per grouping from `varcorr` — `Λ_gΛ_g′ = D_g/σ̂²` — by a Cholesky
/// that zeroes a non-positive pivot rather than failing, so a boundary fit
/// (`θ_d = 0`, `Λ` singular) is handled: the direction contributes nothing and
/// `b̂` is zero there, which is exactly what a pinned component means.
///
/// `Z` is built here in the public per-grouping level-major layout, so this
/// shares no layout knowledge with the kernel's internal RE-column order.
fn brute_ranef(
    fit: &Fit,
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    // Per grouping: per-row level ids, and the x-column of each slope term
    // (empty = intercept-only). Declaration order.
    groups: &[(Vec<usize>, Vec<usize>)],
) -> Vec<f64> {
    let sigma_sq = fit.dispersion;
    // Column count and per-grouping offsets, matching `Fit::ranef`.
    let mut offs = Vec::new();
    let mut k = 0usize;
    for (g, (ids, slopes)) in groups.iter().enumerate() {
        offs.push(k);
        k += fit.ranef_levels[g] * (1 + slopes.len());
        assert_eq!(ids.len(), n);
    }
    assert_eq!(k, fit.ranef.len(), "oracle and fit disagree on ranef width");

    // Z (n×k, row-major) in the public layout.
    let mut z = vec![0.0f64; n * k];
    for (g, (ids, slopes)) in groups.iter().enumerate() {
        let q = 1 + slopes.len();
        for i in 0..n {
            let base = offs[g] + ids[i] * q;
            z[i * k + base] = 1.0;
            for (d, &sc) in slopes.iter().enumerate() {
                z[i * k + base + 1 + d] = x[i * p + sc];
            }
        }
    }

    // Λ (k×k, block-diagonal, one q×q lower factor repeated over the grouping's
    // levels) from each grouping's vech-packed covariance.
    let mut lam = vec![0.0f64; k * k];
    for (g, (_, slopes)) in groups.iter().enumerate() {
        let q = 1 + slopes.len();
        let vech = &fit.varcorr[g];
        let idx = |r: usize, c: usize| c * q - (c * c - c) / 2 + (r - c);
        // D_g/σ̂², symmetric q×q, then its Cholesky.
        let mut d = vec![0.0f64; q * q];
        for c in 0..q {
            for r in c..q {
                let v = vech[idx(r, c)] / sigma_sq;
                d[r * q + c] = v;
                d[c * q + r] = v;
            }
        }
        let mut l = vec![0.0f64; q * q];
        for c in 0..q {
            let mut piv = d[c * q + c];
            for t in 0..c {
                piv -= l[c * q + t] * l[c * q + t];
            }
            if piv <= 0.0 {
                continue; // pinned component: this whole column is zero
            }
            let lc = piv.sqrt();
            l[c * q + c] = lc;
            for r in (c + 1)..q {
                let mut v = d[r * q + c];
                for t in 0..c {
                    v -= l[r * q + t] * l[c * q + t];
                }
                l[r * q + c] = v / lc;
            }
        }
        for lvl in 0..fit.ranef_levels[g] {
            let base = offs[g] + lvl * q;
            for c in 0..q {
                for r in c..q {
                    lam[(base + r) * k + (base + c)] = l[r * q + c];
                }
            }
        }
    }

    // r = y − Xβ̂; A = Λ′Z′ZΛ + I; rhs = Λ′Z′r.
    let resid: Vec<f64> = (0..n)
        .map(|i| y[i] - (0..p).map(|j| x[i * p + j] * fit.beta[j]).sum::<f64>())
        .collect();
    let zl: Vec<f64> = {
        let mut out = vec![0.0f64; n * k];
        for i in 0..n {
            for c in 0..k {
                let mut acc = 0.0;
                for r in c..k {
                    acc += z[i * k + r] * lam[r * k + c];
                }
                out[i * k + c] = acc;
            }
        }
        out
    };
    let mut a = vec![0.0f64; k * k];
    let mut rhs = vec![0.0f64; k];
    for i in 0..n {
        for r in 0..k {
            let zr = zl[i * k + r];
            if zr == 0.0 {
                continue;
            }
            rhs[r] += zr * resid[i];
            for c in 0..k {
                a[r * k + c] += zr * zl[i * k + c];
            }
        }
    }
    for r in 0..k {
        a[r * k + r] += 1.0;
    }
    // Dense Cholesky solve (A is PD: Λ′Z′ZΛ is PSD plus I).
    let mut l = vec![0.0f64; k * k];
    for c in 0..k {
        let mut piv = a[c * k + c];
        for t in 0..c {
            piv -= l[c * k + t] * l[c * k + t];
        }
        assert!(piv > 0.0, "oracle: A is not positive definite");
        let lc = piv.sqrt();
        l[c * k + c] = lc;
        for r in (c + 1)..k {
            let mut v = a[r * k + c];
            for t in 0..c {
                v -= l[r * k + t] * l[c * k + t];
            }
            l[r * k + c] = v / lc;
        }
    }
    let mut u = vec![0.0f64; k];
    for r in 0..k {
        let mut acc = rhs[r];
        for t in 0..r {
            acc -= l[r * k + t] * u[t];
        }
        u[r] = acc / l[r * k + r];
    }
    for r in (0..k).rev() {
        let mut acc = u[r];
        for t in (r + 1)..k {
            acc -= l[t * k + r] * u[t];
        }
        u[r] = acc / l[r * k + r];
    }
    // b = Λu.
    (0..k)
        .map(|r| (0..=r).map(|c| lam[r * k + c] * u[c]).sum())
        .collect()
}

/// Band for the oracle comparison. The two routes solve the SAME system by
/// different eliminations, so they agree to conditioning, not to the last bit;
/// 1e-8 relative to the block's own scale is orders tighter than any
/// disagreement that would indicate a real bug and orders looser than the
/// reassociation floor.
fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-8 * scale,
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

/// `fitted` recomputed the long way: `Xβ̂ + Zb̂` with `Z` written out.
fn assert_fitted(
    fit: &Fit,
    x: &[f64],
    n: usize,
    p: usize,
    groups: &[(Vec<usize>, Vec<usize>)],
    offset: Option<&[f64]>,
) {
    let mut offs = Vec::new();
    let mut k = 0usize;
    for (g, (_, slopes)) in groups.iter().enumerate() {
        offs.push(k);
        k += fit.ranef_levels[g] * (1 + slopes.len());
    }
    for i in 0..n {
        let mut eta = offset.map_or(0.0, |o| o[i]);
        for j in 0..p {
            eta += x[i * p + j] * fit.beta[j];
        }
        for (g, (ids, slopes)) in groups.iter().enumerate() {
            let q = 1 + slopes.len();
            let base = offs[g] + ids[i] * q;
            eta += fit.ranef[base];
            for (d, &sc) in slopes.iter().enumerate() {
                eta += fit.ranef[base + 1 + d] * x[i * p + sc];
            }
        }
        assert!(
            (fit.fitted[i] - eta).abs() <= 1e-9 * eta.abs().max(1.0),
            "fitted[{i}]: got {}, want {eta}",
            fit.fitted[i]
        );
    }
}

/// Lower + fit, returning everything the assertions need.
fn run(formula: &str, table: &Table) -> (Fit, Vec<RanefBlock>, usize, usize, Vec<f64>) {
    let lo = lower(formula, table, Family::Gaussian).expect("lowers");
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged(), "{formula}: did not converge");
    let blocks = label_ranef(&fit, &lo.re_groups).expect("labels");
    (fit, blocks, lo.n, lo.p, lo.x)
}

// ── The recovery, against the oracle, on every route ─────────────────────────

/// Balanced intercept-only `(1|g)`: equal cluster sizes arm the kernel's
/// balanced-collapse path, which replaces the per-family loop with one
/// representative `A(θ)` and a θ-independent Gram combine — the path most likely
/// to be quietly wrong, because it is the one that never forms the per-family
/// couplings the others reuse.
#[test]
fn balanced_collapse_ranef_matches_brute_force() {
    let (x1, x2, y, lab) = sim(48, 42, &["c1", "c2", "c3", "c4", "c5", "c6"]);
    let ids: Vec<usize> = (0..48).map(|i| i % 6).collect();
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y.clone())),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
            ("g".into(), Column::factor_from_labels(&lab)),
        ],
        n: 48,
    };
    let (fit, _, n, p, x) = run("y ~ x1 + x2 + (1 | g)", &table);
    let groups = vec![(ids, vec![])];
    let want = brute_ranef(&fit, &x, &y, n, p, &groups);
    assert_close(&fit.ranef, &want, "collapse ranef");
    assert_fitted(&fit, &x, n, p, &groups, None);
}

/// Unbalanced cluster sizes disarm the collapse, so the general dense path runs
/// its per-family loop — and a random slope makes the family block `q_p×q_p`
/// rather than scalar, exercising the `Λ_p`-folded recovery.
#[test]
fn general_dense_slope_ranef_matches_brute_force() {
    // 47 rows over 6 groups: sizes 8,8,8,8,8,7 — not balanced.
    let (x1, x2, y, lab) = sim(47, 7, &["c1", "c2", "c3", "c4", "c5", "c6"]);
    let ids: Vec<usize> = (0..47).map(|i| i % 6).collect();
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y.clone())),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
            ("g".into(), Column::factor_from_labels(&lab)),
        ],
        n: 47,
    };
    let (fit, _, n, p, x) = run("y ~ x1 + x2 + (1 + x1 | g)", &table);
    // x1 is design column 1 (intercept, x1, x2).
    let groups = vec![(ids, vec![1usize])];
    let want = brute_ranef(&fit, &x, &y, n, p, &groups);
    assert_close(&fit.ranef, &want, "dense slope ranef");
    assert_fitted(&fit, &x, n, p, &groups, None);
}

/// A crossed extra grouping puts levels in the dense tail, which the recovery
/// reaches through a different block of the factor than the families.
#[test]
fn crossed_extra_ranef_matches_brute_force() {
    let n = 60;
    let mut st = 11u64;
    let (mut x1, mut y, mut ga, mut gb) = (vec![], vec![], vec![], vec![]);
    let ua: Vec<f64> = (0..5).map(|_| 0.7 * lcg(&mut st)).collect();
    let ub: Vec<f64> = (0..4).map(|_| 0.5 * lcg(&mut st)).collect();
    for i in 0..n {
        let (a, b) = (i % 5, i % 4);
        let v = lcg(&mut st);
        x1.push(v);
        y.push(0.3 + 0.6 * v + ua[a] + ub[b] + 0.5 * lcg(&mut st));
        ga.push(format!("a{a}"));
        gb.push(format!("b{b}"));
    }
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y.clone())),
            ("x1".into(), numeric(x1)),
            ("ga".into(), Column::factor_from_labels(&ga)),
            ("gb".into(), Column::factor_from_labels(&gb)),
        ],
        n,
    };
    let (fit, _, n, p, x) = run("y ~ x1 + (1 | ga) + (1 | gb)", &table);
    let groups = vec![
        ((0..n).map(|i| i % 5).collect::<Vec<_>>(), vec![]),
        ((0..n).map(|i| i % 4).collect::<Vec<_>>(), vec![]),
    ];
    let want = brute_ranef(&fit, &x, &y, n, p, &groups);
    assert_close(&fit.ranef, &want, "crossed ranef");
    assert_fitted(&fit, &x, n, p, &groups, None);
}

/// A random slope on an EXTRA grouping routes the whole fit to the sparse-Z
/// solver (`fit::classify_design`), whose recovery keeps a different set of
/// factors than the dense one.
#[test]
fn sparse_route_ranef_matches_brute_force() {
    let n = 72;
    let mut st = 99u64;
    let (mut x1, mut y, mut ga, mut gb) = (vec![], vec![], vec![], vec![]);
    let ua: Vec<f64> = (0..6).map(|_| 0.7 * lcg(&mut st)).collect();
    let ub: Vec<f64> = (0..4).map(|_| 0.5 * lcg(&mut st)).collect();
    for i in 0..n {
        let (a, b) = (i % 6, i % 4);
        let v = lcg(&mut st);
        x1.push(v);
        y.push(0.3 + 0.6 * v + ua[a] + ub[b] * (1.0 + 0.3 * v) + 0.5 * lcg(&mut st));
        ga.push(format!("a{a}"));
        gb.push(format!("b{b}"));
    }
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y.clone())),
            ("x1".into(), numeric(x1)),
            ("ga".into(), Column::factor_from_labels(&ga)),
            ("gb".into(), Column::factor_from_labels(&gb)),
        ],
        n,
    };
    let (fit, _, n, p, x) = run("y ~ x1 + (1 | ga) + (1 + x1 | gb)", &table);
    let groups = vec![
        ((0..n).map(|i| i % 6).collect::<Vec<_>>(), vec![]),
        ((0..n).map(|i| i % 4).collect::<Vec<_>>(), vec![1usize]),
    ];
    let want = brute_ranef(&fit, &x, &y, n, p, &groups);
    assert_close(&fit.ranef, &want, "sparse ranef");
    assert_fitted(&fit, &x, n, p, &groups, None);
}

/// The offset is applied to the LMM as an exact `y − o` shift before the
/// sufficient statistics are accumulated, so `fitted` has to add it back — the
/// caveat OLS already documents. Without that, every fitted value is off by the
/// offset and nothing else in the fit notices.
#[test]
fn fitted_adds_the_offset_back() {
    let (x1, x2, y, lab) = sim(48, 5, &["c1", "c2", "c3", "c4"]);
    let off: Vec<f64> = (0..48).map(|i| 0.1 * (i as f64)).collect();
    let y_off: Vec<f64> = y.iter().zip(&off).map(|(a, b)| a + b).collect();
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y_off)),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
            ("g".into(), Column::factor_from_labels(&lab)),
        ],
        n: 48,
    };
    let lo = {
        let mut lo = lower("y ~ x1 + x2 + (1 | g)", &table, Family::Gaussian).unwrap();
        lo.opts.offset = Some(off.clone());
        lo
    };
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged());
    let groups = vec![((0..48).map(|i| i % 4).collect::<Vec<_>>(), vec![])];
    assert_fitted(&fit, &lo.x, lo.n, lo.p, &groups, Some(&off));
}

// ── The labelling, checked as names ──────────────────────────────────────────

/// A plain grouping whose declared level order is NOT sorted. A value-only
/// comparison passes straight through a naming bug, so this asserts the labels
/// themselves — and asserts they follow the DECLARED order, since that is what
/// the ids follow.
#[test]
fn plain_grouping_labels_follow_declared_order() {
    let (x1, x2, y, _) = sim(48, 3, &["z", "a", "m"]);
    let lab: Vec<&str> = (0..48).map(|i| ["z", "a", "m"][i % 3]).collect();
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
            ("g".into(), factor(&["z", "a", "m"], &lab)),
        ],
        n: 48,
    };
    let (_, blocks, _, _, _) = run("y ~ x1 + x2 + (1 | g)", &table);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].group, "g");
    assert_eq!(blocks[0].terms, vec!["(Intercept)".to_string()]);
    assert_eq!(blocks[0].levels, strs(&["z", "a", "m"]));
    assert_eq!(blocks[0].values.len(), 3);
}

/// A random slope makes the block two columns wide; the term names must be the
/// design's own, and each row must carry both its values.
#[test]
fn slope_block_is_levels_by_terms() {
    let (x1, x2, y, lab) = sim(60, 8, &["c1", "c2", "c3", "c4", "c5"]);
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
            ("g".into(), Column::factor_from_labels(&lab)),
        ],
        n: 60,
    };
    let (_, blocks, _, _, _) = run("y ~ x1 + x2 + (1 + x1 | g)", &table);
    assert_eq!(blocks[0].terms, strs(&["(Intercept)", "x1"]));
    assert_eq!(blocks[0].levels.len(), 5);
    assert_eq!(blocks[0].values.len(), 10);
}

/// A crossed interaction grouping `(1|A:B)`: every observed pair is its own
/// level, labelled with the joined name in lexicographic order.
#[test]
fn crossed_interaction_labels_are_joined_pairs() {
    let n = 48;
    let mut st = 21u64;
    let (mut x1, mut y, mut ga, mut gb) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let v = lcg(&mut st);
        x1.push(v);
        y.push(0.2 + 0.5 * v + lcg(&mut st));
        ga.push(format!("a{}", i % 2));
        gb.push(format!("b{}", i % 3));
    }
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("ga".into(), Column::factor_from_labels(&ga)),
            ("gb".into(), Column::factor_from_labels(&gb)),
        ],
        n,
    };
    let (_, blocks, _, _, _) = run("y ~ x1 + (1 | ga:gb)", &table);
    assert_eq!(blocks[0].group, "ga:gb");
    assert_eq!(
        blocks[0].levels,
        strs(&["a0:b0", "a0:b1", "a0:b2", "a1:b0", "a1:b1", "a1:b2"])
    );
}

/// Explicit nesting `(1|A/B)` on an UNBALANCED design: the kernel lays the
/// children out as a padded per-parent rectangle, and `label_ranef` must drop
/// exactly the padded slots. The expected layout is hand-computed, not read back
/// from the implementation: parents A/B/C hold 1/2/3 distinct children, so the
/// rectangle is 3 wide and slots 1, 2 (A's) and 5 (B's) are padding — 6 levels
/// survive out of 9 slots.
#[test]
fn nested_padding_is_dropped_at_the_hand_computed_slots() {
    // 6 distinct (parent, child) cells, repeated to give each some rows.
    let cells: [(&str, &str); 6] = [
        ("A", "c1"),
        ("B", "c1"),
        ("B", "c2"),
        ("C", "c1"),
        ("C", "c2"),
        ("C", "c3"),
    ];
    let n = 60;
    let mut st = 77u64;
    let (mut x1, mut y, mut ga, mut gb) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let (pa, ch) = cells[i % 6];
        let v = lcg(&mut st);
        x1.push(v);
        y.push(0.2 + 0.5 * v + lcg(&mut st));
        ga.push(pa.to_string());
        gb.push(ch.to_string());
    }
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("ga".into(), Column::factor_from_labels(&ga)),
            ("gb".into(), Column::factor_from_labels(&gb)),
        ],
        n,
    };
    let (fit, blocks, _, _, _) = run("y ~ x1 + (1 | ga/gb)", &table);
    // The kernel's own block is the full 3×3 rectangle...
    assert_eq!(fit.ranef_levels[1], 9, "padded rectangle width");
    // ...and the labelled form keeps only the 6 assigned slots, parent-first.
    let inner = &blocks[1];
    assert_eq!(
        inner.levels,
        strs(&["A:c1", "B:c1", "B:c2", "C:c1", "C:c2", "C:c3"])
    );
    assert_eq!(inner.values.len(), 6);
    // The padded slots hold a zero mode by construction — the reason dropping
    // them loses nothing.
    for slot in [1usize, 2, 5] {
        assert_eq!(
            fit.ranef[fit.ranef_levels[0] + slot],
            0.0,
            "padded slot {slot} should hold a zero mode"
        );
    }
}

/// **The case `label_ranef` exists for.** A FLAT pair `(1|A) + (1|B)` with no
/// `:` anywhere in the formula reaches the same padded nested layout as
/// `(1|A/B)` above, purely because the data happens to nest and stay inside the
/// padding bound. A consumer that inferred the layout from the formula would
/// mislabel every row here; the labels must still come out one per observed
/// child, spelled for the child alone (the block is named for the child, so
/// joining the parent in would make the spelling move with the dataset).
#[test]
fn flat_nesting_labels_the_padded_layout_it_silently_routed_to() {
    let cells: [(&str, &str); 6] = [
        ("A", "s1"),
        ("A", "s2"),
        ("B", "s3"),
        ("B", "s4"),
        ("C", "s5"),
        ("C", "s6"),
    ];
    let n = 60;
    let mut st = 31u64;
    let (mut x1, mut y, mut ga, mut gb) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let (pa, ch) = cells[i % 6];
        let v = lcg(&mut st);
        x1.push(v);
        y.push(0.2 + 0.5 * v + lcg(&mut st));
        ga.push(pa.to_string());
        gb.push(ch.to_string());
    }
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("ga".into(), Column::factor_from_labels(&ga)),
            ("gb".into(), Column::factor_from_labels(&gb)),
        ],
        n,
    };
    let (fit, blocks, _, _, _) = run("y ~ x1 + (1 | ga) + (1 | gb)", &table);
    // Balanced 2-per-parent nesting: 3 parents × 2 children, no padding at all,
    // and the block is named for the child alone.
    assert_eq!(fit.ranef_levels[1], 6);
    assert_eq!(blocks[1].group, "gb");
    assert_eq!(
        blocks[1].levels,
        strs(&["s1", "s2", "s3", "s4", "s5", "s6"])
    );
}

/// A declared grouping level between two observed ones costs random-effect
/// width, so it is reported — labelled, with its mode fully shrunk to zero —
/// and the lowering raises a note naming it. lme4 simply has no such row.
#[test]
fn unused_grouping_level_gets_a_zero_row_and_a_note() {
    let lab: Vec<&str> = (0..48).map(|i| ["g1", "g3"][i % 2]).collect();
    let (x1, x2, y, _) = sim(48, 13, &["g1", "g3"]);
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
            // g2 is declared BETWEEN two observed levels, so it owns a slot.
            ("g".into(), factor(&["g1", "g2", "g3"], &lab)),
        ],
        n: 48,
    };
    let lo = lower("y ~ x1 + x2 + (1 | g)", &table, Family::Gaussian).unwrap();
    match lo.notes.as_slice() {
        [Note::UnusedGroupingLevels { grouping, levels }] => {
            assert_eq!(grouping, "g");
            assert_eq!(levels, &strs(&["g2"]));
        }
        other => panic!("expected one unused-levels note, got {other:?}"),
    }
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged());
    let blocks = label_ranef(&fit, &lo.re_groups).unwrap();
    assert_eq!(blocks[0].levels, strs(&["g1", "g2", "g3"]));
    assert_eq!(blocks[0].values[1], 0.0, "an empty cluster is fully shrunk");
}

/// A level declared AFTER the last observed one owns no slot (the block is
/// `max(code)+1` wide), so it is neither labelled nor noted — and the label
/// count still has to match the block, which is what `label_ranef` refuses to
/// guess about.
#[test]
fn trailing_unused_level_costs_nothing_and_is_not_reported() {
    let lab: Vec<&str> = (0..48).map(|i| ["g1", "g2"][i % 2]).collect();
    let (x1, x2, y, _) = sim(48, 17, &["g1", "g2"]);
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
            ("g".into(), factor(&["g1", "g2", "g3"], &lab)),
        ],
        n: 48,
    };
    let lo = lower("y ~ x1 + x2 + (1 | g)", &table, Family::Gaussian).unwrap();
    assert!(lo.notes.is_empty(), "a trailing level costs no width");
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let blocks = label_ranef(&fit, &lo.re_groups).unwrap();
    assert_eq!(blocks[0].levels, strs(&["g1", "g2"]));
}

/// A fixed-effects-only fit has no conditional modes, and labelling one is an
/// empty result rather than an error.
#[test]
fn no_random_effects_labels_to_nothing() {
    let (x1, x2, y, _) = sim(30, 4, &["a"]);
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
        ],
        n: 30,
    };
    let lo = lower("y ~ x1 + x2", &table, Family::Gaussian).unwrap();
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.ranef.is_empty());
    assert!(label_ranef(&fit, &lo.re_groups).unwrap().is_empty());
}

/// `label_ranef` refuses a `re_groups` that does not describe the fit rather
/// than returning a partial result — a mislabelled mode reads as a wrong
/// answer, not a cosmetic slip.
#[test]
fn label_ranef_refuses_a_mismatched_lowering() {
    let (x1, x2, y, lab) = sim(48, 23, &["c1", "c2", "c3"]);
    let table = Table {
        columns: vec![
            ("y".into(), numeric(y)),
            ("x1".into(), numeric(x1)),
            ("x2".into(), numeric(x2)),
            ("g".into(), Column::factor_from_labels(&lab)),
        ],
        n: 48,
    };
    let lo = lower("y ~ x1 + x2 + (1 | g)", &table, Family::Gaussian).unwrap();
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(label_ranef(&fit, &[]).is_err(), "grouping count disagrees");
}
