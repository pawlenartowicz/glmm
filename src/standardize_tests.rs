use super::ColumnScaling;
use faer::Mat;

/// Row-major n×p from rows.
fn rows(r: &[&[f64]]) -> (Vec<f64>, usize, usize) {
    let p = r[0].len();
    (
        r.iter().flat_map(|row| row.iter().copied()).collect(),
        r.len(),
        p,
    )
}

/// One `ColumnScaling::compute` case: the design, its prior weights and
/// random-effect slope columns, and the statistics it must produce.
struct ComputeCase {
    name: &'static str,
    x: (Vec<f64>, usize, usize),
    weights: Option<Vec<f64>>,
    re_slopes: Vec<usize>,
    standardize: bool,
    intercept: Option<usize>,
    /// `(mⱼ, sⱼ)` per column.
    stats: Vec<(f64, f64)>,
    /// Absolute tolerance on `mⱼ` and `sⱼ`: 0 where the value is exact.
    tol: f64,
    identity: bool,
}

/// `√(Σwᵢ(xᵢ − m)²/Σwᵢ)`, the spread `compute` documents, for one column.
fn spread(col: &[f64], w: &[f64], m: f64) -> f64 {
    let sw: f64 = w.iter().sum();
    (col.iter()
        .zip(w)
        .map(|(x, w)| w * (x - m).powi(2))
        .sum::<f64>()
        / sw)
        .sqrt()
}

#[test]
fn compute_finds_the_statistics_the_module_doc_states() {
    let ones = [1.0; 10];
    let weighted = |w: &[f64]| {
        let m = (1.0 * w[0] + 2.0 * w[1] + 6.0 * w[2]) / w.iter().sum::<f64>();
        ComputeCase {
            name: "statistics read the prior weights, not their scale",
            x: rows(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 6.0]]),
            weights: Some(w.to_vec()),
            re_slopes: vec![],
            standardize: true,
            intercept: Some(0),
            stats: vec![(0.0, 1.0), (23.0 / 6.0, spread(&[1.0, 2.0, 6.0], w, m))],
            tol: 1e-14,
            identity: false,
        }
    };
    // 0/1 column 1 and constant column 3 read as slopes; 0/1 column 2 only by
    // the fixed effects. Σwx²/Σw: 7/8 on column 1, 72/8 on column 3.
    let slope_case = |standardize: bool| ComputeCase {
        name: "a 0/1 or constant slope column is scaled, not centered",
        x: rows(&[
            &[1.0, 0.0, 0.0, 3.0],
            &[1.0, 1.0, 1.0, 3.0],
            &[1.0, 1.0, 0.0, 3.0],
        ]),
        weights: Some(vec![1.0, 2.0, 5.0]),
        re_slopes: vec![1, 3],
        standardize,
        intercept: Some(0),
        stats: vec![
            (0.0, 1.0),
            (0.0, (7.0f64 / 8.0).sqrt()),
            (0.0, 1.0),
            (0.0, 3.0),
        ],
        tol: 0.0,
        identity: false,
    };
    let cases = [
        ComputeCase {
            name: "intercept and 0/1 columns are left as they are",
            x: rows(&[&[1.0, 0.0, 2.0], &[1.0, 1.0, 4.0], &[1.0, 1.0, 9.0]]),
            weights: None,
            re_slopes: vec![],
            standardize: true,
            intercept: Some(0),
            stats: vec![
                (0.0, 1.0),
                (0.0, 1.0),
                (5.0, spread(&[2.0, 4.0, 9.0], &ones[..3], 5.0)),
            ],
            tol: 1e-15,
            identity: false,
        },
        ComputeCase {
            name: "the first all-ones column is the intercept, a second is left as is",
            x: rows(&[&[3.0, 1.0, 1.0], &[5.0, 1.0, 1.0], &[4.0, 1.0, 1.0]]),
            weights: None,
            re_slopes: vec![],
            standardize: true,
            intercept: Some(1),
            stats: vec![
                (4.0, spread(&[3.0, 5.0, 4.0], &ones[..3], 4.0)),
                (0.0, 1.0),
                (0.0, 1.0),
            ],
            tol: 1e-15,
            identity: false,
        },
        ComputeCase {
            name: "without an intercept nothing is centered and the scale is the RMS",
            x: rows(&[&[2.0, 0.5], &[4.0, 1.5], &[6.0, 3.0]]),
            weights: None,
            re_slopes: vec![],
            standardize: true,
            intercept: None,
            stats: vec![
                (0.0, spread(&[2.0, 4.0, 6.0], &ones[..3], 0.0)),
                (0.0, spread(&[0.5, 1.5, 3.0], &ones[..3], 0.0)),
            ],
            tol: 1e-15,
            identity: false,
        },
        ComputeCase {
            name: "a constant non-0/1 column next to an intercept is left as it is",
            x: rows(&[&[1.0, 7.0], &[1.0, 7.0], &[1.0, 7.0]]),
            weights: None,
            re_slopes: vec![],
            standardize: true,
            intercept: Some(0),
            stats: vec![(0.0, 1.0), (0.0, 1.0)],
            tol: 0.0,
            identity: true,
        },
        // 0.1 ten times sums to 0.9999999999999999: the spread is round-off,
        // not data.
        ComputeCase {
            name: "a constant column is left as it is when its mean rounds",
            x: ((0..10).flat_map(|_| [1.0, 0.1]).collect(), 10, 2),
            weights: None,
            re_slopes: vec![],
            standardize: true,
            intercept: Some(0),
            stats: vec![(0.0, 1.0), (0.0, 1.0)],
            tol: 0.0,
            identity: true,
        },
        // The loop tier does not screen x.
        ComputeCase {
            name: "a non-finite column is left as it is",
            x: rows(&[&[1.0, f64::NAN], &[1.0, 2.0]]),
            weights: None,
            re_slopes: vec![],
            standardize: true,
            intercept: Some(0),
            stats: vec![(0.0, 1.0), (0.0, 1.0)],
            tol: 0.0,
            identity: true,
        },
        ComputeCase {
            name: "an empty design is the identity",
            x: (vec![], 0, 2),
            weights: None,
            re_slopes: vec![],
            standardize: true,
            intercept: None,
            stats: vec![(0.0, 1.0), (0.0, 1.0)],
            tol: 0.0,
            identity: true,
        },
        weighted(&[1.0, 2.0, 3.0]),
        weighted(&[10.0, 20.0, 30.0]),
        slope_case(true),
        // The slopes are scaled the same way when the design is fitted as given.
        slope_case(false),
    ];
    for c in cases {
        let (x, n, p) = &c.x;
        let mut cs = ColumnScaling::with_re_slopes(*p, c.re_slopes.iter().copied());
        cs.compute(x, *n, *p, c.weights.as_deref(), c.standardize);
        assert_eq!(cs.intercept(), c.intercept, "{}", c.name);
        assert_eq!(cs.is_identity(), c.identity, "{}", c.name);
        for (j, &(m, s)) in c.stats.iter().enumerate() {
            let got = (cs.center(j), cs.scale(j));
            assert!(
                (got.0 - m).abs() <= c.tol && (got.1 - s).abs() <= c.tol,
                "{}: column {j} (m, s) = {got:?}, want ({m}, {s})",
                c.name
            );
        }
    }
}

fn fixture() -> (ColumnScaling, usize) {
    // Intercept in column 1 (not 0) on purpose, one centered column, one binary.
    let (x, n, p) = rows(&[
        &[3.0, 1.0, 0.0],
        &[5.0, 1.0, 1.0],
        &[10.0, 1.0, 1.0],
        &[2.0, 1.0, 0.0],
    ]);
    let mut cs = ColumnScaling::identity(p);
    cs.compute(&x, n, p, None, true);
    (cs, p)
}

/// The β maps and the covariance congruence are `B` of the module doc: β = Bβ̃
/// both ways round, and Cov(β) = B·Cov(β̃)·Bᵀ.
#[test]
fn beta_and_covariance_maps_are_b() {
    let (map, p) = fixture();
    let bd = Mat::from_fn(p, p, |t, i| map.rhs(t, i));
    let bt = [0.7, -1.3, 0.4];
    let mut b = bt;
    map.beta_to_caller(&mut b);
    for t in 0..p {
        let want: f64 = (0..p).map(|i| bd[(t, i)] * bt[i]).sum();
        assert!((b[t] - want).abs() < 1e-14, "row {t}");
        assert!((map.beta_coef(t, &bt) - want).abs() < 1e-14);
    }
    map.beta_to_internal(&mut b);
    for t in 0..p {
        assert!((b[t] - bt[t]).abs() < 1e-14);
    }
    let m0 = Mat::from_fn(p, p, |a, b| if a == b { 2.0 + a as f64 } else { 0.3 });
    let mut m = m0.clone();
    map.congruence(m.as_mut());
    let want = &bd * &m0 * bd.transpose();
    for a in 0..p {
        for b in 0..p {
            assert!((m[(a, b)] - want[(a, b)]).abs() < 1e-13, "({a},{b})");
        }
    }
}

#[test]
fn column_standardization_is_in_place() {
    let (x, n, p) = rows(&[&[1.0, 2.0], &[1.0, 4.0], &[1.0, 9.0]]);
    let mut cs = ColumnScaling::identity(p);
    cs.compute(&x, n, p, None, true);
    for j in 0..p {
        let mut col: Vec<f64> = (0..n).map(|i| x[i * p + j]).collect();
        cs.standardize_column(j, &mut col);
        for i in 0..n {
            assert_eq!(col[i], (x[i * p + j] - cs.center(j)) / cs.scale(j));
        }
    }
}
