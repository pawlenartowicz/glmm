// The robust-fit checks at the kernel level: the pieces
// against the frozen CR2 / HC2 references, the pole sum against the dense
// route (a port of `validation/tools/cr2/lib.py::dense_cluster`), and against
// exact identities.

use super::*;
use crate::inference::rlrt_refs::csv_columns;
use crate::inference::test_refs::{entries, load, Kind};
use std::collections::BTreeMap;

/// One cluster's data, owned.
struct Cl {
    x: Mat<f64>,
    y: Vec<f64>,
    lam: Vec<f64>,
    u: Mat<f64>,
    r: Vec<f64>,
}

fn blocks(cl: &[Cl]) -> Vec<ClusterBlock<'_>> {
    cl.iter()
        .map(|c| ClusterBlock {
            x: c.x.as_ref(),
            r: &c.r,
            lambda: &c.lam,
            u: c.u.as_ref(),
        })
        .collect()
}

/// GLS at the clusters' Φ: fills each `r`, returns M and R = Lᵀ, L the
/// Cholesky factor of M⁻¹ (a fit's route: M⁻¹ factored, never M inverted
/// back).
fn gls(cl: &mut [Cl]) -> (Mat<f64>, Mat<f64>) {
    let p = cl[0].x.ncols();
    let mut info = Mat::<f64>::zeros(p, p);
    let mut g = vec![0.0; p];
    for c in cl.iter() {
        let xy = Mat::from_fn(
            c.y.len(),
            p + 1,
            |i, j| if j < p { c.x[(i, j)] } else { c.y[i] },
        );
        let t = phi_cross(&c.lam, c.u.as_ref(), xy.as_ref());
        for j in 0..p {
            g[j] += t[(j, p)];
            for i in 0..p {
                info[(i, j)] += t[(i, j)];
            }
        }
    }
    let m = sym(solve(info.as_ref(), Mat::<f64>::identity(p, p).as_ref()).as_ref());
    let beta: Vec<f64> = (0..p)
        .map(|i| (0..p).map(|j| m[(i, j)] * g[j]).sum())
        .collect();
    for c in cl.iter_mut() {
        c.r = (0..c.y.len())
            .map(|i| c.y[i] - (0..p).map(|j| c.x[(i, j)] * beta[j]).sum::<f64>())
            .collect();
    }
    let r = SeqLlt::new(info.as_ref())
        .unwrap()
        .L()
        .transpose()
        .to_owned();
    (m, r)
}

fn unit(p: usize, j: usize) -> Vec<f64> {
    (0..p).map(|i| f64::from(u8::from(i == j))).collect()
}

/// SE and df of each coefficient.
fn se_df(pc: &Cr2Pieces) -> (Vec<f64>, Vec<f64>) {
    let p = pc.t.nrows();
    let v = pc.vcov();
    let se = (0..p).map(|j| v[(j, j)].sqrt()).collect();
    let df = (0..p).map(|j| pc.df(&unit(p, j))).collect();
    (se, df)
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x / y - 1.0).abs())
        .fold(0.0, f64::max)
}

fn floats(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

/// The clusters of a `tools/prep/gen_cr2_data.py` CSV: X = [1, x1, x2], Φ_k =
/// diag(lam) + U Uᵀ with U the `u` columns; without `g`, one row per cluster
/// and Φ = diag(1 / w) (the HC2 sets).
fn clusters(path: &str) -> Vec<Cl> {
    let d = csv_columns(path);
    let n = d["y"].len();
    let ucols: Vec<&Vec<f64>> = d
        .iter()
        .filter(|(k, _)| k.starts_with('u'))
        .map(|(_, v)| v)
        .collect();
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let g = d.get("g").map_or(i as i64, |g| g[i] as i64);
        groups.entry(g).or_default().push(i);
    }
    groups
        .values()
        .map(|rows| Cl {
            x: Mat::from_fn(rows.len(), 3, |i, j| match j {
                0 => 1.0,
                1 => d["x1"][rows[i]],
                _ => d["x2"][rows[i]],
            }),
            y: rows.iter().map(|&i| d["y"][i]).collect(),
            lam: rows
                .iter()
                .map(|&i| d.get("lam").map_or(1.0 / d["w"][i], |l| l[i]))
                .collect(),
            u: Mat::from_fn(rows.len(), ucols.len(), |i, j| ucols[j][rows[i]]),
            r: vec![],
        })
        .collect()
}

/// Checks the frozen SE and df (and the vcov when the reference has one) of
/// every `kind` entry `generator` wrote, to `tol` relative; returns the worst gap.
fn golden_gap(kind: Kind, generator: &str, tol: f64) -> f64 {
    let mut worst = 0.0f64;
    let list: Vec<_> = entries(kind)
        .into_iter()
        .filter(|e| e.generator == generator)
        .collect();
    assert!(!list.is_empty(), "no {generator} entry");
    for e in list {
        let reference = load(kind, &e.name);
        let mut cl = clusters(e.data.as_deref().unwrap());
        let (_, r) = gls(&mut cl);
        let pc = cr2_pieces(&blocks(&cl), r.as_ref()).unwrap();
        let (se, df) = se_df(&pc);
        // Dense at any size where the conditioning allows it, which
        // sends the spread-1e4 and 1e6 clusters to the pole sum, agrees with it.
        let (se_p, df_p) = se_df(&pole_pieces(&cl, &r));
        let any_size = cr2_pieces_with(&blocks(&cl), r.as_ref(), (usize::MAX, DENSE.1)).unwrap();
        let (se_d, df_d) = se_df(&any_size);
        let routes = max_rel(&se_p, &se_d).max(max_rel(&df_p, &df_d));
        assert!(routes <= 1e-12, "{}: routes differ by {routes:.2e}", e.name);
        let mut gap =
            max_rel(&se, &floats(&reference["se"])).max(max_rel(&df, &floats(&reference["df"])));
        if let Some(rows) = reference.get("vcov") {
            let v = pc.vcov();
            let r: Vec<Vec<f64>> = rows.as_array().unwrap().iter().map(floats).collect();
            let scale = r.iter().flatten().fold(0.0f64, |a, b| a.max(b.abs()));
            for (i, row) in r.iter().enumerate() {
                for (j, x) in row.iter().enumerate() {
                    gap = gap.max((v[(i, j)] - x).abs() / scale);
                }
            }
        }
        eprintln!("{}: worst relative gap {gap:.2e}", e.name);
        assert!(gap <= tol, "{}: gap {gap:.2e} above {tol:.0e}", e.name);
        worst = worst.max(gap);
    }
    worst
}

#[test]
fn g2_dense_references_at_spread_up_to_100() {
    golden_gap(Kind::Cr2, "goldens_cr2.R", 1e-9);
}

#[test]
fn g3_high_precision_references_at_spread_up_to_1e6() {
    golden_gap(Kind::Cr2Hp, "goldens_cr2_hp.py", 1e-9);
}

#[test]
fn g4_hc2_matches_club_sandwich() {
    golden_gap(Kind::Cr2, "goldens_hc2.R", 1e-10);
}

#[test]
fn hc2_record_is_linear_in_n() {
    let mut cl = clusters("data/inference/hc2_n100.csv");
    let (_, r) = gls(&mut cl);
    let pc = cr2_pieces(&blocks(&cl), r.as_ref()).unwrap();
    assert!(pc.whole.is_empty());
    assert_eq!((pc.t.nrows(), pc.t.ncols()), (3, 3));
    assert_eq!((pc.er.nrows(), pc.er.ncols()), (3, 100));
    assert_eq!((pc.row_e.nrows(), pc.row_e.ncols()), (3, 100));
    assert_eq!((pc.row_x.nrows(), pc.row_x.ncols()), (3, 100));
    assert_eq!(pc.row_phi.len(), 100);
}

/// The CR2 entry of the `cr2_df` references (12 clusters of 10, Φ_k = I +
/// τ²11ᵀ, two columns 1e-7 apart) against its 50-digit df. R is the R factor
/// of a QR of the stacked L_k⁻¹X_k, L_k the Cholesky factor of Φ_k: the route
/// of a fit's Givens factor on a nearly singular design.
#[test]
fn cr2_df_match_50_digits_on_a_nearly_collinear_design() {
    let list: Vec<_> = entries(Kind::Cr2Df)
        .into_iter()
        .filter(|e| e.raw.get("cluster").is_some())
        .collect();
    assert_eq!(list.len(), 1, "cr2_df entries with clusters");
    let e = &list[0];
    let d = csv_columns(e.data.as_deref().unwrap());
    let tau = e.raw["tau2"].as_f64().unwrap().sqrt();
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, g) in d[e.raw["cluster"].as_str().unwrap()].iter().enumerate() {
        groups.entry(*g as i64).or_default().push(i);
    }
    let cols = ["a", "xi", "xi2"];
    let cl: Vec<Cl> = groups
        .values()
        .map(|rows| Cl {
            x: Mat::from_fn(rows.len(), 4, |i, j| match j {
                0 => 1.0,
                _ => d[cols[j - 1]][rows[i]],
            }),
            y: vec![],
            lam: vec![1.0; rows.len()],
            u: Mat::from_fn(rows.len(), 1, |_, _| tau),
            // The df does not read the residuals.
            r: rows.iter().map(|&i| d["y"][i]).collect(),
        })
        .collect();
    let n: usize = cl.iter().map(|c| c.r.len()).sum();
    let mut stack = Mat::<f64>::zeros(n, 4);
    let mut at = 0;
    for c in &cl {
        let nk = c.r.len();
        let phi = Mat::from_fn(nk, nk, |i, j| f64::from(u8::from(i == j)) + tau * tau);
        let llt = SeqLlt::new(phi.as_ref()).unwrap();
        let l = llt.L();
        for j in 0..4 {
            for i in 0..nk {
                let acc: f64 = (0..i).map(|k| l[(i, k)] * stack[(at + k, j)]).sum();
                stack[(at + i, j)] = (c.x[(i, j)] - acc) / l[(i, i)];
            }
        }
        at += nk;
    }
    let r = stack.as_ref().qr().thin_R().to_owned();
    let pc = cr2_pieces(&blocks(&cl), r.as_ref()).unwrap();
    let df: Vec<f64> = (0..4).map(|j| pc.df(&unit(4, j))).collect();
    let want = floats(&load(Kind::Cr2Df, &e.name)["df"]);
    assert_eq!((df.len(), want.len()), (4, 4));
    let gap = max_rel(&df, &want);
    assert!(gap <= 1e-8, "df {df:?} against {want:?}: {gap:.2e}");
}

// ---- the two routes ----

/// The pieces with every multi-row cluster on the pole sum.
fn pole_pieces(cl: &[Cl], r: &Mat<f64>) -> Cr2Pieces {
    cr2_pieces_with(&blocks(cl), r.as_ref(), (0, 0.0)).unwrap()
}

/// The pieces with every multi-row cluster on the dense route, a port of
/// `lib.py::dense_cluster`, whatever its conditioning.
fn dense_pieces(cl: &[Cl], r: &Mat<f64>) -> Cr2Pieces {
    cr2_pieces_with(&blocks(cl), r.as_ref(), (usize::MAX, f64::INFINITY)).unwrap()
}

/// Worst relative gap of SE and df over the unit contrasts, kernel vs dense.
fn gap_to_dense(cl: &[Cl], r: &Mat<f64>) -> (f64, f64) {
    let (se_q, df_q) = se_df(&pole_pieces(cl, r));
    let (se_d, df_d) = se_df(&dense_pieces(cl, r));
    (max_rel(&se_q, &se_d), max_rel(&df_q, &df_d))
}

// ---- seeded data ----

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

/// `rank_check.py`'s `build`: X = [1, x1, x2], weights log-uniform over a
/// spread of 100, Z = [1] or [1, z] with z = x1 (`slope = 1`) or the
/// cluster-level x2 (`slope = 2`), U = Z L with L L' = `re`.
fn build(rng: &mut Rng, nks: &[usize], re: [[f64; 2]; 2], q: usize, slope: usize) -> Vec<Cl> {
    // Cholesky of re + 1e-300 I, as rank_check.py; zero when re is zero.
    let any = re.iter().flatten().any(|&v| v != 0.0);
    let l00 = if any { (re[0][0] + 1e-300).sqrt() } else { 0.0 };
    let (l10, l11) = if q == 2 && any {
        let l10 = re[1][0] / l00;
        (l10, (re[1][1] + 1e-300 - l10 * l10).sqrt())
    } else {
        (0.0, 0.0)
    };
    let x2: Vec<f64> = nks.iter().map(|_| rng.normal()).collect();
    nks.iter()
        .zip(&x2)
        .map(|(&n, &x2k)| {
            let x1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let lw: Vec<f64> = (0..n).map(|_| rng.uniform() * 100f64.ln()).collect();
            let mean = lw.iter().sum::<f64>() / n as f64;
            let lam: Vec<f64> = lw.iter().map(|v| 1.0 / (v - mean).exp()).collect();
            let z = |i: usize| if slope == 1 { x1[i] } else { x2k };
            let u = Mat::from_fn(n, q, |i, j| match (q, j) {
                (1, _) => l00,
                (_, 0) => l00 + z(i) * l10,
                _ => z(i) * l11,
            });
            let b: Vec<f64> = (0..q).map(|_| rng.normal()).collect();
            let y = (0..n)
                .map(|i| {
                    1.0 + 0.5 * x1[i]
                        + 0.3 * x2k
                        + (0..q).map(|j| u[(i, j)] * b[j]).sum::<f64>()
                        + rng.normal() * lam[i].sqrt()
                })
                .collect();
            Cl {
                x: Mat::from_fn(n, 3, |i, j| [1.0, x1[i], x2k][j]),
                y,
                lam,
                u,
                r: vec![],
            }
        })
        .collect()
}

#[test]
fn g7_rank_of_u_matches_dense() {
    let slope = [[1.0, 0.2], [0.2, 0.25]];
    let mut worst = 0.0f64;
    for seed in 1..=3u64 {
        let mut rng = Rng(seed);
        let small: Vec<usize> = (0..200)
            .map(|_| 2 + (rng.uniform() * 4.0) as usize)
            .collect();
        let cases: Vec<(&str, Vec<Cl>)> = vec![
            (
                "zero RE variance",
                build(&mut rng, &[40; 10], [[0.0; 2]; 2], 1, 0),
            ),
            (
                "RE variance 1e-20",
                build(&mut rng, &[40; 10], [[1e-20, 0.0], [0.0, 0.0]], 1, 0),
            ),
            (
                "zero slope variance",
                build(&mut rng, &[40; 10], [[1.0, 0.0], [0.0, 0.0]], 2, 1),
            ),
            (
                "slope constant in cluster",
                build(&mut rng, &[40; 10], slope, 2, 2),
            ),
            (
                "clusters of 1 and 2, slope",
                build(&mut rng, &[1, 2, 1, 2, 30, 30, 30, 30, 30, 30], slope, 2, 1),
            ),
            (
                "many clusters of 2 to 5",
                build(&mut rng, &small, slope, 2, 1),
            ),
        ];
        for (name, mut cl) in cases {
            for c in cl.iter_mut() {
                c.u = reduce_u(&c.lam, c.u.as_ref()).unwrap();
            }
            let (_, r) = gls(&mut cl);
            let (se, df) = gap_to_dense(&cl, &r);
            eprintln!("seed {seed} {name:28}: SE {se:.2e} df {df:.2e}");
            worst = worst.max(se).max(df);
        }
    }
    eprintln!("rank of U, worst gap to dense {worst:.2e}");
    assert!(worst <= 4.2e-13, "worst gap {worst:.2e}");
}

/// Six clusters of eight rows, random intercept; X = [1, x1, t] with t = 1 in
/// cluster 0 only (one null direction) and, with `two`, also t·x1 (two).
fn leverage_one(two: bool) -> Vec<Cl> {
    let mut rng = Rng(11);
    (0..6)
        .map(|k| {
            let n = 8;
            let t = f64::from(u8::from(k == 0));
            let x1: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let lam: Vec<f64> = (0..n).map(|_| (rng.uniform() * 10f64.ln()).exp()).collect();
            let b = rng.normal() * 0.7;
            let y = (0..n)
                .map(|i| 1.0 + 0.5 * x1[i] + b + rng.normal() * lam[i].sqrt())
                .collect();
            let p = if two { 4 } else { 3 };
            Cl {
                x: Mat::from_fn(n, p, |i, j| [1.0, x1[i], t, t * x1[i]][j]),
                y,
                lam,
                u: Mat::from_fn(n, 1, |_, _| 0.7),
                r: vec![],
            }
        })
        .collect()
}

/// Worst relative gap of the pieces (per piece and cluster, to the dense
/// piece's largest entry) and of the within contrasts' SE and df.
fn leverage_gap(cl: &[Cl], r: &Mat<f64>, d: usize) -> f64 {
    let p = r.nrows();
    let pc = pole_pieces(cl, r);
    assert_eq!(pc.null.len(), d);
    assert_eq!(dense_pieces(cl, r).null.len(), d);
    assert!(pc.null.iter().all(|(k, _)| *k == 0));
    let dn = dense_pieces(cl, r);
    let mut gap = 0.0f64;
    for ((f, epe), (fd, epd)) in pc.whole.iter().zip(&dn.whole) {
        for (a, b) in [(f, fd), (epe, epd)] {
            let scale = (0..p)
                .flat_map(|j| (0..p).map(move |i| (i, j)))
                .fold(0.0f64, |s, (i, j)| s.max(b[(i, j)].abs()));
            for j in 0..p {
                for i in 0..p {
                    gap = gap.max((a[(i, j)] - b[(i, j)]).abs() / scale);
                }
            }
        }
    }
    for k in 0..cl.len() {
        let scale = (0..p).fold(0.0f64, |s, i| s.max(dn.er[(i, k)].abs()));
        for i in 0..p {
            gap = gap.max((pc.er[(i, k)] - dn.er[(i, k)]).abs() / scale);
        }
    }
    // the within contrasts (intercept, x1) keep their CR2 SE and df
    let (se, df) = se_df(&pc);
    let (sed, dfd) = se_df(&dn);
    gap.max(max_rel(&se[..2], &sed[..2]))
        .max(max_rel(&df[..2], &dfd[..2]))
}

#[test]
fn g7_leverage_one_clusters_match_dense() {
    for (two, d) in [(false, 1), (true, 2)] {
        let mut cl = leverage_one(two);
        let (_, r) = gls(&mut cl);
        let gap = leverage_gap(&cl, &r, d);
        eprintln!("leverage 1, {d} null direction(s): worst gap to dense {gap:.2e}");
        assert!(gap <= 2e-13, "gap {gap:.2e}");
    }
}

#[test]
fn g7_one_cluster_contrast_is_flagged_and_a_within_one_is_not() {
    let mut cl = leverage_one(true);
    let (_, r) = gls(&mut cl);
    let pc = cr2_pieces(&blocks(&cl), r.as_ref()).unwrap();
    for j in [2, 3] {
        assert!(pc.identified_by_one(&unit(4, j)), "column {j}");
    }
    for j in [0, 1] {
        assert!(!pc.identified_by_one(&unit(4, j)), "column {j}");
    }
    assert!(pc.identified_by_one(&[0.0, 1.0, 0.0, 1.0]));
    let mut ri = clusters("data/inference/cr2_ri_s10.csv");
    let (_, r) = gls(&mut ri);
    let pc = cr2_pieces(&blocks(&ri), r.as_ref()).unwrap();
    assert!(pc.null.is_empty());
    assert!(!pc.identified_by_one(&unit(3, 2)));
}

#[test]
fn g7_trace_q_is_the_gls_variance() {
    for path in [
        "data/inference/cr2_rs_s10.csv",
        "data/inference/hc2_n20.csv",
    ] {
        let mut cl = clusters(path);
        let (m, r) = gls(&mut cl);
        let pc = cr2_pieces(&blocks(&cl), r.as_ref()).unwrap();
        for c in [unit(3, 0), unit(3, 1), unit(3, 2), vec![0.3, -1.2, 0.8]] {
            let (tr, _) = pc.q_traces(&pc.whiten(&c));
            let cmc: f64 = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| c[i] * m[(i, j)] * c[j])
                .sum();
            assert!(
                (tr / cmc - 1.0).abs() <= 1e-12,
                "{path}: tr Q {tr} vs {cmc}"
            );
        }
    }
}

/// SE and df after scaling: r by `sr`, Φ and M by `sv` (R by 1/√`sv`).
fn scaled(cl: &[Cl], r: &Mat<f64>, sr: f64, sv: f64) -> (Vec<f64>, Vec<f64>) {
    let cl2: Vec<Cl> = cl
        .iter()
        .map(|c| Cl {
            x: c.x.clone(),
            y: vec![],
            lam: c.lam.iter().map(|l| l * sv).collect(),
            u: Mat::from_fn(c.u.nrows(), c.u.ncols(), |i, j| c.u[(i, j)] * sv.sqrt()),
            r: c.r.iter().map(|r| r * sr).collect(),
        })
        .collect();
    let r2 = Mat::from_fn(r.nrows(), r.ncols(), |i, j| r[(i, j)] / sv.sqrt());
    se_df(&cr2_pieces(&blocks(&cl2), r2.as_ref()).unwrap())
}

#[test]
fn g7_y_and_w_scales_leave_the_df_and_scale_the_se() {
    for path in [
        "data/inference/cr2_rs_s10.csv",
        "data/inference/hc2_n20.csv",
    ] {
        let mut cl = clusters(path);
        let (_, r) = gls(&mut cl);
        let (se, df) = scaled(&cl, &r, 1.0, 1.0);
        // y · 1e-4: r · 1e-4, and σ² (so Φ and M in the caller's units) · 1e-8
        let (se_y, df_y) = scaled(&cl, &r, 1e-4, 1e-8);
        let se_y: Vec<f64> = se_y.iter().map(|s| s / 1e-4).collect();
        assert!(
            max_rel(&se_y, &se) <= 1e-12 && max_rel(&df_y, &df) <= 1e-12,
            "{path}: y scale"
        );
        // w · 1e7: Φ and M in the kernel's units (σ² = 1) · 1e-7
        let (se_w, df_w) = scaled(&cl, &r, 1.0, 1e-7);
        assert!(
            max_rel(&se_w, &se) <= 1e-12 && max_rel(&df_w, &df) <= 1e-12,
            "{path}: w scale"
        );
    }
}

// ---- the identity A C A z = Φ z on a cluster no dense code can hold ----

/// Re[(Φ − i s)⁻¹ Z] through the split Z = Z̃ + U B (Z̃ᵀΛ⁻¹U = 0), with
/// Ω = (Λ − i s)⁻¹ and K = I + UᵀΩU: (Φ − i s)⁻¹ U = Ω U K⁻¹ and
/// (Φ − i s)⁻¹ Z̃ = Ω Z̃ − Ω U K⁻¹ Uᵀ Ω Z̃.
fn r_inv(lam: &[f64], u: &Mat<f64>, s: f64, z: MatRef<'_, f64>) -> Mat<f64> {
    let (n, q, k) = (lam.len(), u.ncols(), z.ncols());
    let (zt, bz, _) = split(lam, u.as_ref(), z);
    let om: Vec<c64> = lam.iter().map(|&l| c64::new(l, -s).inv()).collect();
    let omu = Mat::from_fn(n, q, |i, j| om[i] * u[(i, j)]);
    let omz = Mat::from_fn(n, k, |i, j| om[i] * zt[(i, j)]);
    let uc = Mat::from_fn(n, q, |i, j| c64::new(u[(i, j)], 0.0));
    let kmat = Mat::from_fn(q, q, |i, j| {
        (0..n).map(|r| uc[(r, i)] * omu[(r, j)]).sum::<c64>()
            + if i == j {
                c64::new(1.0, 0.0)
            } else {
                c64::new(0.0, 0.0)
            }
    });
    // rhs = Uᵀ Ω Z̃ − B
    let rhs = Mat::from_fn(q, k, |i, j| {
        (0..n).map(|r| uc[(r, i)] * omz[(r, j)]).sum::<c64>() - c64::new(bz[(i, j)], 0.0)
    });
    let sol = solve(kmat.as_ref(), rhs.as_ref());
    Mat::from_fn(n, k, |i, j| {
        (omz[(i, j)] - (0..q).map(|t| omu[(i, t)] * sol[(t, j)]).sum::<c64>()).re
    })
}

/// A z by the kernel's pole sum: z + Σ_j w_j R_j⁻¹ X G_j⁻¹ Xᵀ R_j⁻¹ z.
struct Apply<'a> {
    c: &'a Cl,
    u: Mat<f64>,
    // per pole: weight, R⁻¹X, G⁻¹
    poles: Vec<(f64, Mat<f64>, Mat<f64>)>,
}

impl Apply<'_> {
    fn new<'a>(c: &'a Cl, hmk: &Mat<f64>, m: &Mat<f64>) -> Apply<'a> {
        let u = reduce_u(&c.lam, c.u.as_ref()).unwrap();
        let h = sym(phi_cross(&c.lam, u.as_ref(), c.x.as_ref()).as_ref());
        let lm = SeqLlt::new(m.as_ref()).unwrap().L().to_owned();
        let sp = spectrum(&c.lam, &u, hmk, lm.as_ref()).unwrap();
        assert_eq!(sp.vnull.ncols(), 0);
        let (s, w) = nodes(sp.zmin, sp.zmax, n_poles(sp.zmax / sp.zmin));
        let poles = s
            .iter()
            .zip(&w)
            .map(|(&sj, &wj)| {
                let rx = r_inv(&c.lam, &u, sj, c.x.as_ref());
                let q = mul(c.x.transpose(), rx.as_ref());
                let p = h.nrows();
                let g = Mat::from_fn(p, p, |i, j| {
                    hmk[(i, j)] + h[(i, j)] - 0.5 * (q[(i, j)] + q[(j, i)])
                });
                (
                    wj,
                    rx,
                    solve(g.as_ref(), Mat::<f64>::identity(p, p).as_ref()),
                )
            })
            .collect();
        Apply { c, u, poles }
    }
    fn a(&self, z: &Mat<f64>) -> Mat<f64> {
        let mut out = z.clone();
        for (w, rx, gi) in &self.poles {
            let t = mul(gi.as_ref(), mul(rx.transpose(), z.as_ref()).as_ref());
            let add = mul(rx.as_ref(), t.as_ref());
            for j in 0..z.ncols() {
                for i in 0..z.nrows() {
                    out[(i, j)] += w * add[(i, j)];
                }
            }
        }
        out
    }
    fn phi(&self, z: &Mat<f64>) -> Mat<f64> {
        let uz = mul(
            self.u.as_ref(),
            mul(self.u.transpose(), z.as_ref()).as_ref(),
        );
        Mat::from_fn(z.nrows(), z.ncols(), |i, j| {
            self.c.lam[i] * z[(i, j)] + uz[(i, j)]
        })
    }
}

/// The worst relative gap of A C A z = Φ z over the columns z of [Φ⁻¹X, r],
/// the vectors the fit applies A to.
fn aca_gap(c: &Cl, hmk: &Mat<f64>, m: &Mat<f64>) -> (f64, usize) {
    let ap = Apply::new(c, hmk, m);
    let (n, p) = (c.lam.len(), m.nrows());
    let px = phi_solve(&c.lam, ap.u.as_ref(), c.x.as_ref());
    let z = Mat::from_fn(n, p + 1, |i, j| if j < p { px[(i, j)] } else { c.r[i] });
    let az = ap.a(&z);
    // C y = Φ y − X M Xᵀ y
    let xmxty = mul(
        mul(c.x.as_ref(), m.as_ref()).as_ref(),
        mul(c.x.transpose(), az.as_ref()).as_ref(),
    );
    let phiaz = ap.phi(&az);
    let acaz = ap.a(&Mat::from_fn(n, p + 1, |i, j| {
        phiaz[(i, j)] - xmxty[(i, j)]
    }));
    let phiz = ap.phi(&z);
    let gap = (0..=p)
        .map(|j| {
            let num: f64 = (0..n).map(|i| (acaz[(i, j)] - phiz[(i, j)]).powi(2)).sum();
            let den: f64 = (0..n).map(|i| phiz[(i, j)].powi(2)).sum();
            (num / den).sqrt()
        })
        .fold(0.0, f64::max);
    (gap, ap.poles.len())
}

#[test]
#[ignore = "a cluster of 100,000 rows; run by name with --ignored"]
fn g6_aca_is_phi_at_100k_rows() {
    let start = std::time::Instant::now();
    let slope = [[1.0, 0.2], [0.2, 0.25]];
    let mut cl = build(&mut Rng(5), &[100_000, 100, 100, 100], slope, 2, 1);
    let (m, _) = gls(&mut cl);
    let mut hmk = Mat::<f64>::zeros(3, 3);
    for c in &cl[1..] {
        let u = reduce_u(&c.lam, c.u.as_ref()).unwrap();
        add_scaled(
            &mut hmk,
            &sym(phi_cross(&c.lam, u.as_ref(), c.x.as_ref()).as_ref()),
            1.0,
        );
    }
    let (gap, n) = aca_gap(&cl[0], &hmk, &m);
    eprintln!(
        "A C A z = Φ z at n_k = 100,000: {gap:.2e} with {n} poles, {:.1} s",
        start.elapsed().as_secs_f64()
    );
    assert!(gap <= 4e-13 * 10.0, "gap {gap:.2e}");
}
