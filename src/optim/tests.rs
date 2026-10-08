use super::bfgs::{bfgs_minimize, BfgsConfig, BfgsState, BfgsStop};
use super::{bobyqa_minimize, GradObjective, Objective};
use bobyqa::{Bobyqa, Config};

/// A closure pair as a [`GradObjective`]: the gradient is recomputed at the
/// last value point, which is what a state-reading objective returns.
struct Closed<F, G> {
    f: F,
    g: G,
    last: Vec<f64>,
}

impl<F: FnMut(&[f64]) -> f64, G: FnMut(&[f64], &mut [f64])> Objective for Closed<F, G> {
    fn value(&mut self, x: &[f64], _resolution: f64) -> Option<f64> {
        self.last.clear();
        self.last.extend_from_slice(x);
        Some((self.f)(x))
    }
}

impl<F: FnMut(&[f64]) -> f64, G: FnMut(&[f64], &mut [f64])> GradObjective for Closed<F, G> {
    fn gradient(&mut self, g: &mut [f64]) -> bool {
        (self.g)(&self.last, g);
        true
    }
}

fn closed<F, G>(f: F, g: G) -> Closed<F, G> {
    Closed {
        f,
        g,
        last: Vec::new(),
    }
}

#[test]
fn bfgs_solves_a_quadratic_and_certifies() {
    // f = ½ xᵀAx − bᵀx with A SPD; optimum A⁻¹b = (1, −2, 0.5).
    let a = [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0];
    let xs = [1.0, -2.0, 0.5];
    let b: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| a[i * 3 + j] * xs[j]).sum())
        .collect();
    let f = |x: &[f64]| {
        let mut v = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                v += 0.5 * x[i] * a[i * 3 + j] * x[j];
            }
            v -= b[i] * x[i];
        }
        v
    };
    let g = |x: &[f64], g: &mut [f64]| {
        for i in 0..3 {
            g[i] = (0..3).map(|j| a[i * 3 + j] * x[j]).sum::<f64>() - b[i];
        }
    };
    let mut obj = closed(f, g);
    let mut x = [0.3, 0.3, 0.3];
    let mut st = BfgsState::default();
    let out = bfgs_minimize(
        &mut obj,
        &mut x,
        &[-10.0; 3],
        &[10.0; 3],
        &BfgsConfig::LMM,
        &mut st,
        None,
    );
    // Converged in f to the `Fdelta` floor (1e-9), so x to about
    // √(1e-9 / curvature).
    for i in 0..3 {
        assert!((x[i] - xs[i]).abs() < 1e-4, "x={x:?}");
    }
    assert!(out.converged(), "{out:?}");
    assert!(out.certified(&x, &[0, 2]), "{out:?}");
}

#[test]
fn bfgs_reaches_the_rosenbrock_minimum() {
    let f = |x: &[f64]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
    let g = |x: &[f64], g: &mut [f64]| {
        g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
    };
    let mut obj = closed(f, g);
    let mut x = [-1.2, 1.0];
    let mut st = BfgsState::default();
    let out = bfgs_minimize(
        &mut obj,
        &mut x,
        &[-5.0; 2],
        &[5.0; 2],
        &BfgsConfig::LMM,
        &mut st,
        None,
    );
    assert!(
        (x[0] - 1.0).abs() < 1e-4 && (x[1] - 1.0).abs() < 1e-4,
        "{x:?} {out:?}"
    );
    assert!(out.f < 1e-8, "{out:?}");
}

#[test]
fn bfgs_stops_on_an_active_bound() {
    // Unconstrained optimum (3, −2) lies outside [−1, 1]²; the box optimum is
    // the corner (1, −1), where both coordinates are held.
    let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
    let g = |x: &[f64], g: &mut [f64]| {
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 2.0 * (x[1] + 2.0);
    };
    let mut obj = closed(f, g);
    let mut x = [0.0, 0.0];
    let mut st = BfgsState::default();
    let out = bfgs_minimize(
        &mut obj,
        &mut x,
        &[-1.0; 2],
        &[1.0; 2],
        &BfgsConfig::LMM,
        &mut st,
        None,
    );
    assert_eq!(x, [1.0, -1.0], "{out:?}");
    assert_eq!(out.stop, BfgsStop::Pgrad);
    assert_eq!(out.decrement, 0.0);
}

/// Contract C1: the gradient is asked for only right after a full-tolerance
/// value call that returned a finite value, and that point is where the
/// search goes next (or the start).
#[test]
fn bfgs_asks_for_the_gradient_only_at_the_last_finite_value() {
    #[derive(Default)]
    struct Recorder {
        last: Vec<f64>,
        last_finite: bool,
        calls: Vec<(char, Vec<f64>)>,
    }
    impl Objective for Recorder {
        fn value(&mut self, x: &[f64], resolution: f64) -> Option<f64> {
            assert_eq!(resolution, 0.0, "BFGS evaluates at full tolerance");
            self.last = x.to_vec();
            // A wall at x0 > 0.8 makes some trials fail.
            let v = if x[0] > 0.8 {
                f64::INFINITY
            } else {
                (x[0] - 2.0).powi(2) + 3.0 * (x[1] - 0.5).powi(2)
            };
            self.last_finite = v.is_finite();
            self.calls.push(('v', x.to_vec()));
            Some(v)
        }
    }
    impl GradObjective for Recorder {
        fn gradient(&mut self, g: &mut [f64]) -> bool {
            assert!(self.last_finite, "gradient after a failed value");
            assert!(
                matches!(self.calls.last(), Some(('v', _))),
                "gradient not right after a value call"
            );
            g[0] = 2.0 * (self.last[0] - 2.0);
            g[1] = 6.0 * (self.last[1] - 0.5);
            self.calls.push(('g', self.last.clone()));
            true
        }
    }
    let mut obj = Recorder::default();
    let mut x = [0.0, 0.0];
    let mut st = BfgsState::default();
    let out = bfgs_minimize(
        &mut obj,
        &mut x,
        &[-5.0; 2],
        &[5.0; 2],
        &BfgsConfig::LMM,
        &mut st,
        None,
    );
    assert!(out.n_grad >= 2, "{out:?}");
    let last_g = obj.calls.iter().rev().find(|c| c.0 == 'g').unwrap();
    assert_eq!(
        last_g.1,
        x.to_vec(),
        "the incumbent is the last gradient point"
    );
    assert_eq!(out.n_val, obj.calls.iter().filter(|c| c.0 == 'v').count());
}

/// The search keeps nothing between runs: a second run on the same state is
/// bit-identical to a run on a fresh state.
#[test]
fn bfgs_state_reuse_is_bit_identical() {
    let f = |x: &[f64]| (x[0] - 1.0).powi(4) + (x[0] * x[1] - 2.0).powi(2) + x[2] * x[2];
    let g = |x: &[f64], g: &mut [f64]| {
        g[0] = 4.0 * (x[0] - 1.0).powi(3) + 2.0 * (x[0] * x[1] - 2.0) * x[1];
        g[1] = 2.0 * (x[0] * x[1] - 2.0) * x[0];
        g[2] = 2.0 * x[2];
    };
    let run = |st: &mut BfgsState, x0: [f64; 3]| {
        let mut obj = closed(f, g);
        let mut x = x0;
        let out = bfgs_minimize(
            &mut obj,
            &mut x,
            &[-9.0; 3],
            &[9.0; 3],
            &BfgsConfig::LMM,
            st,
            None,
        );
        (x, out.f, out.n_val, out.n_grad)
    };
    let mut reused = BfgsState::default();
    let _ = run(&mut reused, [3.0, -1.0, 2.0]);
    let a = run(&mut reused, [0.5, 0.5, 0.5]);
    let b = run(&mut BfgsState::default(), [0.5, 0.5, 0.5]);
    assert_eq!(a.0.map(f64::to_bits), b.0.map(f64::to_bits));
    assert_eq!((a.1.to_bits(), a.2, a.3), (b.1.to_bits(), b.2, b.3));
}

#[test]
fn bfgs_reports_a_non_finite_start() {
    let mut obj = closed(|_: &[f64]| f64::NAN, |_: &[f64], _: &mut [f64]| {});
    let mut x = [1.0];
    let out = bfgs_minimize(
        &mut obj,
        &mut x,
        &[-1.0],
        &[2.0],
        &BfgsConfig::LMM,
        &mut BfgsState::default(),
        None,
    );
    assert_eq!(out.stop, BfgsStop::NonFinite);
    assert!(!out.certified(&x, &[]));
}

/// The BOBYQA wrapper runs exactly `Bobyqa::minimize`'s search and hands the
/// objective the trust radius.
#[test]
fn bobyqa_wrapper_matches_minimize_bit_for_bit() {
    let f = |x: &[f64]| (x[0] - 0.3).powi(2) + 2.0 * (x[1] + 0.7).powi(2) + x[0] * x[1];
    let cfg = Config::new(2);
    let mut x_ref = [1.0, 1.0];
    let ref_out = Bobyqa::new(2, cfg)
        .unwrap()
        .minimize(f, &mut x_ref, &[-3.0; 2], &[3.0; 2]);
    struct WithRho<F> {
        f: F,
        rhos: Vec<f64>,
    }
    impl<F: FnMut(&[f64]) -> f64> Objective for WithRho<F> {
        fn value(&mut self, x: &[f64], resolution: f64) -> Option<f64> {
            self.rhos.push(resolution);
            Some((self.f)(x))
        }
    }
    let mut obj = WithRho {
        f,
        rhos: Vec::new(),
    };
    let mut x = [1.0, 1.0];
    let mut solver = Bobyqa::new(2, cfg).unwrap();
    let out = bobyqa_minimize(&mut solver, &mut obj, &mut x, &[-3.0; 2], &[3.0; 2]);
    assert_eq!(x.map(f64::to_bits), x_ref.map(f64::to_bits));
    assert_eq!(out.f.to_bits(), ref_out.f.to_bits());
    assert_eq!(out.n_eval, ref_out.n_eval);
    assert_eq!(obj.rhos.len(), out.n_eval);
    assert_eq!(obj.rhos[0], cfg.rho_begin);
}

/// The layer hosts a 1-D method on the value-only trait: a golden-section
/// search written against [`Objective`] alone, reporting its bracket width as
/// the resolution.
#[test]
fn objective_hosts_a_one_dimensional_method() {
    fn golden(obj: &mut impl Objective, mut a: f64, mut b: f64, tol: f64) -> f64 {
        const R: f64 = 0.618_033_988_749_894_9;
        let mut c = b - R * (b - a);
        let mut d = a + R * (b - a);
        let mut fc = obj.value(&[c], b - a).unwrap_or(f64::INFINITY);
        let mut fd = obj.value(&[d], b - a).unwrap_or(f64::INFINITY);
        while b - a > tol {
            if fc < fd {
                b = d;
                d = c;
                fd = fc;
                c = b - R * (b - a);
                fc = obj.value(&[c], b - a).unwrap_or(f64::INFINITY);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + R * (b - a);
                fd = obj.value(&[d], b - a).unwrap_or(f64::INFINITY);
            }
        }
        0.5 * (a + b)
    }
    struct Quad(Vec<f64>);
    impl Objective for Quad {
        fn value(&mut self, x: &[f64], resolution: f64) -> Option<f64> {
            self.0.push(resolution);
            Some((x[0] - 0.42).powi(2))
        }
    }
    let mut obj = Quad(Vec::new());
    let x = golden(&mut obj, 0.0, 2.0, 1e-8);
    assert!((x - 0.42).abs() < 1e-7);
    assert!(obj.0.windows(2).all(|w| w[1] <= w[0]), "bracket shrinks");
}

/// A hint replaces the identity start: on a quadratic, the exact inverse
/// Hessian takes the search to the optimum in one step.
#[test]
fn bfgs_with_the_exact_inverse_hessian_steps_to_the_optimum() {
    let f = |x: &[f64]| 2.0 * (x[0] - 1.0).powi(2) + 0.5 * (x[1] + 3.0).powi(2);
    let g = |x: &[f64], g: &mut [f64]| {
        g[0] = 4.0 * (x[0] - 1.0);
        g[1] = x[1] + 3.0;
    };
    let h = super::CurvatureHint::from_gradients(&[0.0, 0.0], &[-9.0; 2], &[9.0; 2], |x, gg| {
        g(x, gg);
        true
    })
    .unwrap();
    let mut obj = closed(f, g);
    let mut x = [0.0, 0.0];
    let out = bfgs_minimize(
        &mut obj,
        &mut x,
        &[-9.0; 2],
        &[9.0; 2],
        &BfgsConfig::LMM,
        &mut BfgsState::default(),
        Some(h.inv_hessian()),
    );
    assert!(
        (x[0] - 1.0).abs() < 1e-9 && (x[1] + 3.0).abs() < 1e-9,
        "{x:?} {out:?}"
    );
    assert!(out.iters <= 2, "{out:?}");
}

#[test]
fn curvature_hint_refuses_an_indefinite_hessian() {
    // A saddle: ∂²f = diag(2, −2).
    let h = super::CurvatureHint::from_gradients(&[0.5, 0.5], &[-9.0; 2], &[9.0; 2], |x, g| {
        g[0] = 2.0 * x[0];
        g[1] = -2.0 * x[1];
        true
    });
    assert!(h.is_none());
}
