//! Exact invariances of the fit: a transform of the data whose effect on the
//! model is known in closed form must move the fit by that map and nothing
//! else. Each row fits a base model and its transformed twin through the
//! matrix door and compares the two. The level-recoding row goes through the
//! formula door, where factor levels exist.
//!
//! The data are deterministic (no RNG): n = 40 rows in 8 groups of 5, a
//! second grouping crossed with the first (5 levels), `x` from a fixed
//! sequence, `y` from fixed effects, fixed group effects and a fixed noise
//! sequence, positive for the Gamma and inverse-Gaussian fits. The binomial
//! successes out of 20 trials and the Poisson counts come from the same
//! effects through the logit and log links.

use glmm::inference::ProfileParam;
use glmm::{
    fit, fit_with_start, BinomialLink, Data, Family, Fit, FitOptions, GammaLink, GroupIds,
    Grouping, InverseGaussianLink, ModelSpec, NegBinomialLink, Note, PoissonLink, Re, StartValues,
    WeightPower,
};

const N: usize = 40;
const P: usize = 2;

/// The same optimum: `dev_eps` of `validation/grid/tol.R`, the grid's measured band.
const DEV_EPS: f64 = 4e-5;

/// Relative band on every mapped parameter (β, SEs, `varcorr`, dispersion,
/// fitted values) in every row: where the search may stop elsewhere on a
/// flat optimum, parameters move by a few 1e-6 while the objective agrees to
/// 1e-10 or better.
const TOL: f64 = 1e-5;

fn x_seq(i: usize) -> f64 {
    (1.7 * i as f64 + 0.3).sin()
}

fn noise(i: usize) -> f64 {
    (2.3 * i as f64 + 0.7).sin()
}

/// Row-major `[1, x]`.
fn design() -> Vec<f64> {
    (0..N).flat_map(|i| [1.0, x_seq(i)]).collect()
}

/// Primary grouping `g` (8 levels of 5 rows) and the crossed one `h` (5 levels).
fn ids() -> (Vec<u32>, Vec<u32>) {
    let g = (0..N as u32).map(|i| i / 5).collect();
    let h = (0..N as u32).map(|i| i % 5).collect();
    (g, h)
}

const U_G: [f64; 8] = [-0.6, 0.4, 0.1, -0.3, 0.7, -0.2, 0.5, -0.4];
const A_H: [f64; 5] = [0.5, -0.3, 0.2, -0.6, 0.3];
const V_H: [f64; 5] = [0.4, -0.5, 0.3, -0.2, 0.1];

fn gaussian_y() -> Vec<f64> {
    let (g, h) = ids();
    (0..N)
        .map(|i| {
            let (gi, hi) = (g[i] as usize, h[i] as usize);
            2.0 + 0.5 * x_seq(i) + U_G[gi] + A_H[hi] + V_H[hi] * x_seq(i) + 0.4 * noise(i)
        })
        .collect()
}

/// Positive: `exp(0.5 + 0.3x + 0.5·u_g + 0.3·noise)`.
fn positive_y() -> Vec<f64> {
    let (g, _) = ids();
    (0..N)
        .map(|i| (0.5 + 0.3 * x_seq(i) + 0.5 * U_G[g[i] as usize] + 0.3 * noise(i)).exp())
        .collect()
}

/// Weights in [1, 4], so ×1e-100 and ×1e12 stay inside the weight range.
fn base_weights() -> Vec<f64> {
    (0..N)
        .map(|i| {
            let t = 0.618034 * i as f64 + 0.1;
            1.0 + 3.0 * (t - t.floor())
        })
        .collect()
}

/// Binomial trials per row.
const TRIALS: f64 = 20.0;

/// Random slopes of `h` for the binomial and count responses, nearly
/// uncorrelated with `A_H` so their fits sit inside the parameter space
/// (`V_H` gives a binomial slope fit at |ρ| = 1).
const W_H: [f64; 5] = [-0.3, -0.4, 0.5, 0.4, -0.2];

/// Successes out of `TRIALS`: `round(20·logistic(η))` with
/// `η = 0.3 + 0.8x + 3·u_g + a_h + w_h·x + 0.3·noise`. The `g` effects are
/// large enough that the one-trial fit of `k/20` has a nonzero variance.
fn binomial_k() -> Vec<f64> {
    let (g, h) = ids();
    (0..N)
        .map(|i| {
            let (gi, hi, x) = (g[i] as usize, h[i] as usize, x_seq(i));
            let eta = 0.3 + 0.8 * x + 3.0 * U_G[gi] + A_H[hi] + W_H[hi] * x + 0.3 * noise(i);
            (TRIALS / (1.0 + (-eta).exp())).round()
        })
        .collect()
}

/// The binomial response on the matrix door: the proportion `k/20`.
fn binomial_share() -> Vec<f64> {
    binomial_k().iter().map(|k| k / TRIALS).collect()
}

/// Counts: `round(exp(2 + 0.3x + u_g + a_h + w_h·x + 0.6·noise))`, mean
/// about 9 and overdispersed against Poisson, so the negative binomial shape
/// is inside its range.
fn count_y() -> Vec<f64> {
    let (g, h) = ids();
    (0..N)
        .map(|i| {
            let (gi, hi, x) = (g[i] as usize, h[i] as usize, x_seq(i));
            let eta = 2.0 + 0.3 * x + U_G[gi] + A_H[hi] + W_H[hi] * x + 0.6 * noise(i);
            eta.exp().round()
        })
        .collect()
}

/// 0/1: whether `binomial_k` is at least half its trials.
fn binary_y() -> Vec<f64> {
    binomial_k()
        .iter()
        .map(|&k| if k >= TRIALS / 2.0 { 1.0 } else { 0.0 })
        .collect()
}

fn data<'a>(
    x: &'a [f64],
    y: &'a [f64],
    w: Option<&'a [f64]>,
    ids: Option<&'a GroupIds>,
) -> Data<'a> {
    let mut d = Data::new(x, y, N, P);
    if let Some(w) = w {
        d = d.weights(w);
    }
    if let Some(ids) = ids {
        d = d.ids(ids);
    }
    d
}

/// The deviance, or −2·loglik on the routes that report none (OLS, GLM).
fn objective(f: &Fit) -> f64 {
    if f.deviance().is_finite() {
        f.deviance()
    } else {
        -2.0 * f.loglik()
    }
}

/// Whether the fit tells the user its numbers may be off: a
/// `precision_limited` or `variance_at_limit` note, or the ports'
/// `search_limit` (not converged, yet a finite deviance and every estimated β
/// finite; the rule of `glmm-bind`'s `notes.rs`).
fn flagged(f: &Fit) -> bool {
    let noted = f
        .diagnostics()
        .notes
        .iter()
        .any(|n| matches!(n, Note::PrecisionLimited | Note::VarianceAtLimit));
    let search_limit = !f.converged()
        && f.deviance().is_finite()
        && f.beta()
            .iter()
            .zip(f.aliased())
            .all(|(b, &aliased)| aliased || b.is_finite());
    noted || search_limit
}

/// The counts and failures of one row. A case whose values miss their band
/// passes only when its transformed fit is [`flagged`], and is counted there.
#[derive(Default)]
struct Tally {
    cases: usize,
    compared: usize,
    outside: usize,
    flagged: usize,
    failures: Vec<String>,
}

impl Tally {
    fn case(&mut self, at: String) -> Case<'_> {
        Case {
            tally: self,
            at,
            misses: Vec::new(),
        }
    }

    fn assert_clean(&self) {
        assert!(self.failures.is_empty(), "{}", self.failures.join("\n"));
    }
}

struct Case<'t> {
    tally: &'t mut Tally,
    at: String,
    misses: Vec<String>,
}

impl Case<'_> {
    /// `got` against `want` at `TOL` relative; NaN on both sides is equal
    /// (an output NaN by its docs on both fits).
    fn close(&mut self, what: &str, got: f64, want: f64) {
        self.tally.compared += 1;
        let d = if got.is_nan() && want.is_nan() {
            0.0
        } else {
            rel(got, want)
        };
        if d.is_nan() || d > TOL {
            let at = &self.at;
            self.misses
                .push(format!("{at}: {what} {got:e} vs {want:e} (rel {d:e})"));
        }
    }

    fn close_all(&mut self, what: &str, got: &[f64], want: &[f64]) {
        if got.len() != want.len() {
            let at = &self.at;
            let (a, b) = (got.len(), want.len());
            self.misses
                .push(format!("{at}: {what} has {a} values vs {b}"));
            return;
        }
        for (a, b) in got.iter().zip(want) {
            self.close(what, *a, *b);
        }
    }

    /// A mapped value: compared when it is representable, else counted apart.
    fn mapped(&mut self, what: &str, got: f64, want: f64) {
        if representable(want) {
            self.close(what, got, want);
        } else {
            self.tally.outside += 1;
        }
    }

    fn objective(&mut self, got: f64, want: f64) {
        let d = (got - want).abs();
        if d.is_nan() || d > DEV_EPS {
            let at = &self.at;
            self.misses
                .push(format!("{at}: objective {got:e} vs {want:e} (moved {d:e})"));
        }
    }

    fn same<T: PartialEq + std::fmt::Debug>(&mut self, what: &str, got: T, want: T) {
        if got != want {
            let at = &self.at;
            self.misses
                .push(format!("{at}: {what} {got:?} vs {want:?}"));
        }
    }

    /// A failure no note excuses.
    fn fail(&mut self, what: String) {
        let at = &self.at;
        self.tally.failures.push(format!("{at}: {what}"));
    }

    /// Closes the case on `twin`, the transformed fit.
    fn end(self, twin: &Fit) {
        self.tally.cases += 1;
        if self.misses.is_empty() {
            return;
        }
        if flagged(twin) {
            self.tally.flagged += 1;
        } else {
            self.tally.failures.extend(self.misses);
        }
    }
}

fn assert_clean_base(f: &Fit, name: &str) {
    assert!(f.converged(), "{name}: the base fit converges");
    assert!(!f.singular(), "{name}: the base fit is not singular");
}

struct Model {
    name: &'static str,
    spec: ModelSpec,
    crossed: bool,
    nagq: u8,
}

fn every_varcorr(f: &Fit) -> Vec<f64> {
    (0..f.n_groupings()).flat_map(|g| f.varcorr(g)).collect()
}

/// `w → c·w` on the dispersion families: β, SEs and every `varcorr` entry
/// unchanged, the dispersion times `c` (row `i` has variance `φ·V(μᵢ)/wᵢ`, so
/// `φ` takes the scale the weights lose). At a power-of-two `c` the internal
/// weights `w/s` are the same bits, so the fit is the same bit for bit. At any
/// other `c` they differ by a factor in [1/√2, √2] and the search may stop
/// elsewhere on the same optimum: there the same entries are compared to
/// `TOL` relative, and the objective to `DEV_EPS` and `TOL·max(1, |dev|)`.
#[test]
fn weights_times_c_leave_the_dispersion_families_unchanged() {
    // Largest seen: 4.5e-6 relative (a `varcorr` entry of the Gamma inverse
    // nAGQ 5 fit at c = 1e-12), the log-likelihood equal to 1.6e-10.
    let gamma_log = Family::Gamma {
        link: GammaLink::Log,
    };
    let gamma_inv = Family::Gamma {
        link: GammaLink::Inverse,
    };
    let models = [
        Model {
            name: "gaussian OLS",
            spec: ModelSpec::fixed(Family::Gaussian),
            crossed: false,
            nagq: 1,
        },
        Model {
            name: "gaussian (1 | g)",
            spec: ModelSpec::mixed(Family::Gaussian, Re::intercept()),
            crossed: false,
            nagq: 1,
        },
        Model {
            name: "gaussian (1 | g) + (1 + x | h)",
            spec: ModelSpec::mixed(
                Family::Gaussian,
                Re::intercept().extra(Grouping::crossed().slopes([1])),
            ),
            crossed: true,
            nagq: 1,
        },
        Model {
            name: "gamma log GLM",
            spec: ModelSpec::fixed(gamma_log),
            crossed: false,
            nagq: 1,
        },
        Model {
            name: "gamma inverse GLM",
            spec: ModelSpec::fixed(gamma_inv),
            crossed: false,
            nagq: 1,
        },
        Model {
            name: "gamma log (1 | g) Laplace",
            spec: ModelSpec::mixed(gamma_log, Re::intercept()),
            crossed: false,
            nagq: 1,
        },
        Model {
            name: "gamma log (1 | g) nagq 5",
            spec: ModelSpec::mixed(gamma_log, Re::intercept()),
            crossed: false,
            nagq: 5,
        },
        Model {
            name: "gamma inverse (1 | g) Laplace",
            spec: ModelSpec::mixed(gamma_inv, Re::intercept()),
            crossed: false,
            nagq: 1,
        },
        Model {
            name: "gamma inverse (1 | g) nagq 5",
            spec: ModelSpec::mixed(gamma_inv, Re::intercept()),
            crossed: false,
            nagq: 5,
        },
        Model {
            name: "inverse gaussian log GLM",
            spec: ModelSpec::fixed(Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            }),
            crossed: false,
            nagq: 1,
        },
        Model {
            name: "inverse gaussian 1/mu^2 GLM",
            spec: ModelSpec::fixed(Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            }),
            crossed: false,
            nagq: 1,
        },
    ];
    let x = design();
    let (g, h) = ids();
    let w0 = base_weights();
    let (mut exact_cases, mut tol_cases, mut compared) = (0, 0, 0);
    let mut failures = Vec::new();
    for m in &models {
        let y = if m.spec.family == Family::Gaussian {
            gaussian_y()
        } else {
            positive_y()
        };
        let mut gids = GroupIds::new(g.clone());
        if m.crossed {
            gids = gids.extra(h.clone());
        }
        let mut opts = FitOptions::default();
        opts.nagq = m.nagq;
        if m.spec.family == Family::Gaussian {
            opts.weight_power = Some(WeightPower::Fixed(1.0));
        }
        let fit_at = |c: f64| {
            let w: Vec<f64> = w0.iter().map(|v| c * v).collect();
            let mut data = Data::new(&x, &y, N, P).weights(&w);
            if m.spec.re.is_some() {
                data = data.ids(&gids);
            }
            fit(&data, &m.spec, &opts).unwrap_or_else(|e| panic!("{} at c = {c:e}: {e}", m.name))
        };
        // The deviance, or −2·loglik on the routes that report none (OLS, GLM).
        let objective = |f: &Fit| {
            if f.deviance().is_finite() {
                f.deviance()
            } else {
                -2.0 * f.loglik()
            }
        };
        let base = fit_at(1.0);
        assert!(base.converged(), "{}: the base fit converges", m.name);
        assert!(!base.singular(), "{}: the base fit is not singular", m.name);
        for c in [2f64.powi(-332), 2f64.powi(-40), 1.0, 2f64.powi(40)] {
            let f = fit_at(c);
            let mut same = |what: &str, a: f64, b: f64| {
                if a != b {
                    failures.push(format!("{} at c = {c:e}: {what} {a:e} vs {b:e}", m.name));
                }
            };
            same("converged", f64::from(u8::from(f.converged())), 1.0);
            same("singular", f64::from(u8::from(f.singular())), 0.0);
            for j in 0..P {
                same("beta", f.beta()[j], base.beta()[j]);
                same("se", f.se()[j], base.se()[j]);
            }
            for (a, b) in every_varcorr(&f).iter().zip(every_varcorr(&base)) {
                same("varcorr", *a, b);
            }
            same("dispersion", f.dispersion(), c * base.dispersion());
            exact_cases += 1;
        }
        for c in [1e-100, 1e-12, 1e12] {
            let f = fit_at(c);
            let at = |what: &str| format!("{} at c = {c:e}: {what}", m.name);
            if f.converged() != base.converged() || f.singular() != base.singular() {
                failures.push(at("converged or singular flipped"));
            }
            let d = (objective(&f) - objective(&base)).abs();
            if d.is_nan() || d > DEV_EPS.min(TOL * objective(&base).abs().max(1.0)) {
                failures.push(at(&format!("objective moved by {d:e}")));
            }
            let mut check = |what: &str, a: f64, b: f64| {
                compared += 1;
                let d = rel(a, b);
                if d.is_nan() || d > TOL {
                    failures.push(at(&format!("{what} {a:e} vs {b:e} (rel {d:e})")));
                }
            };
            for j in 0..P {
                check("beta", f.beta()[j], base.beta()[j]);
                check("se", f.se()[j], base.se()[j]);
            }
            for (a, b) in every_varcorr(&f).iter().zip(every_varcorr(&base)) {
                check("varcorr", *a, b);
            }
            check("dispersion", f.dispersion(), c * base.dispersion());
            tol_cases += 1;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!((exact_cases, tol_cases, compared), (44, 33, 195));
}

/// Relative difference `|a − b| / max(|a|, |b|, tiny)`, 0 when both are equal
/// (both zero included).
fn rel(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    (a - b).abs() / a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
}

/// Whether `v` is a normal f64 or zero: a value the map can carry exactly.
fn representable(v: f64) -> bool {
    v == 0.0 || v.is_normal()
}

/// `y → c·y` on Gamma and inverse Gaussian: both are scale families, so the
/// fit on `c·y` is the fit on `y` mapped by the link. With `k` the factor β
/// takes (1 on the log link, where the intercept takes `+ln c` instead; `1/c`
/// on the inverse link; `1/c²` on 1/μ²): β and the SEs times `k`, `vcov`
/// times `k²`, `varcorr` (the random-effect variance) times `k²`, `ranef`
/// times `k`, the dispersion unchanged on Gamma and over `c` on inverse
/// Gaussian, the fitted means times `c`, the log-likelihood `− n·ln c` and
/// the GLMM deviance `+ 2n·ln c`.
///
/// The map is checked entry by entry at every `c`. A mapped value outside the
/// f64 range (or below the normal range), and an SE whose variance is, has
/// no exact value to compare and is counted apart. At a power-of-two `c` the
/// internal response `y/c` is the same bits; at any other `c` it differs by a
/// factor in [1/√2, √2] and the search may stop elsewhere on the same
/// optimum, so the objective is compared there too.
#[test]
fn response_times_c_maps_back_on_gamma_and_inverse_gaussian() {
    // Largest seen: 1.7e-15 relative on the GLMs; 1.5e-6 on the Laplace
    // GLMMs at 1e-200 and 1e80 (a `varcorr` entry of the Gamma log fit at
    // 1e80), where the log-likelihood agrees to 1.8e-12.
    let gamma_log = Family::Gamma {
        link: GammaLink::Log,
    };
    let gamma_inv = Family::Gamma {
        link: GammaLink::Inverse,
    };
    let ig_log = Family::InverseGaussian {
        link: InverseGaussianLink::Log,
    };
    let ig_inv2 = Family::InverseGaussian {
        link: InverseGaussianLink::InverseSquared,
    };
    let models = [
        ("gamma log GLM", ModelSpec::fixed(gamma_log)),
        ("gamma inverse GLM", ModelSpec::fixed(gamma_inv)),
        (
            "gamma log (1 | g) Laplace",
            ModelSpec::mixed(gamma_log, Re::intercept()),
        ),
        (
            "gamma inverse (1 | g) Laplace",
            ModelSpec::mixed(gamma_inv, Re::intercept()),
        ),
        ("inverse gaussian log GLM", ModelSpec::fixed(ig_log)),
        ("inverse gaussian 1/mu^2 GLM", ModelSpec::fixed(ig_inv2)),
    ];
    let x = design();
    let (g, _) = ids();
    let gids = GroupIds::new(g);
    let y0 = positive_y();
    let mut opts = FitOptions::default();
    opts.condvar = true;
    let (mut map_cases, mut objective_cases, mut compared, mut outside) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    for (name, spec) in &models {
        let fit_at = |c: f64| {
            let y: Vec<f64> = y0.iter().map(|v| c * v).collect();
            let mut data = Data::new(&x, &y, N, P);
            if spec.re.is_some() {
                data = data.ids(&gids);
            }
            fit(&data, spec, &opts).unwrap_or_else(|e| panic!("{name} at c = {c:e}: {e}"))
        };
        let objective = |f: &Fit| {
            if f.deviance().is_finite() {
                f.deviance()
            } else {
                -2.0 * f.loglik()
            }
        };
        let base = fit_at(1.0);
        assert!(base.converged(), "{name}: the base fit converges");
        assert!(!base.singular(), "{name}: the base fit is not singular");
        let mixed = spec.re.is_some();
        for c in [
            2f64.powi(-800),
            2f64.powi(-664),
            2f64.powi(266),
            1e-200,
            1e80,
        ] {
            let f = fit_at(c);
            let (k, log) = match spec.family {
                Family::Gamma {
                    link: GammaLink::Inverse,
                } => (1.0 / c, false),
                Family::InverseGaussian {
                    link: InverseGaussianLink::InverseSquared,
                } => ((1.0 / c) * (1.0 / c), false),
                _ => (1.0, true),
            };
            let ig = matches!(spec.family, Family::InverseGaussian { .. });
            if f.converged() != base.converged() {
                failures.push(format!("{name} at c = {c:e}: converged flipped"));
            }
            let mut check = |what: &str, got: f64, want: f64, exact_range: bool| {
                if !(exact_range && representable(want)) {
                    outside += 1;
                    return;
                }
                compared += 1;
                let d = rel(got, want);
                if d.is_nan() || d > TOL {
                    failures.push(format!(
                        "{name} at c = {c:e}: {what} {got:e} vs {want:e} (rel {d:e})"
                    ));
                }
            };
            for j in 0..P {
                let want = if log {
                    base.beta()[j] + if j == 0 { c.ln() } else { 0.0 }
                } else {
                    k * base.beta()[j]
                };
                check("beta", f.beta()[j], want, true);
                let se = k * base.se()[j];
                check("se", f.se()[j], se, representable(se * se));
            }
            for (a, b) in f.vcov().iter().zip(base.vcov()) {
                check("vcov", *a, k * k * b, true);
            }
            for (a, b) in every_varcorr(&f).iter().zip(every_varcorr(&base)) {
                check("varcorr", *a, k * k * b, true);
            }
            if mixed {
                for (a, b) in f.ranef(0).iter().zip(base.ranef(0)) {
                    check("ranef", *a, k * b, true);
                }
                let len = (f.condvar(0).len(), base.condvar(0).len());
                assert_eq!(len.0, len.1, "{name} at c = {c:e}: condvar length");
                for (a, b) in f.condvar(0).iter().zip(base.condvar(0)) {
                    check("condvar", *a, k * k * b, true);
                }
                let dev = base.deviance() + 2.0 * N as f64 * c.ln();
                check("deviance", f.deviance(), dev, true);
            }
            let disp = if ig {
                base.dispersion() / c
            } else {
                base.dispersion()
            };
            check("dispersion", f.dispersion(), disp, true);
            for (a, b) in f.fitted().iter().zip(base.fitted()) {
                check("fitted", *a, c * b, true);
            }
            let ll = base.loglik() - N as f64 * c.ln();
            check("loglik", f.loglik(), ll, true);
            map_cases += 1;
        }
        for c in [1e-200, 1e80] {
            let f = fit_at(c);
            let at = |what: &str| format!("{name} at c = {c:e}: {what}");
            if f.converged() != base.converged() {
                failures.push(at("converged flipped"));
            }
            let want = objective(&base) + 2.0 * N as f64 * c.ln();
            let d = (objective(&f) - want).abs();
            if d.is_nan() || d > DEV_EPS {
                failures.push(at(&format!("objective moved by {d:e}")));
            }
            objective_cases += 1;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!((map_cases, objective_cases), (30, 12));
    // Outside the f64 range: the inverse link's SEs and `vcov` (and the GLMM's
    // `varcorr` and `condvar`) at the three small scales, where the variance passes 1e308;
    // the 1/μ² link's β there too, and its SEs and `vcov` at 2²⁶⁶ and 1e80,
    // where the variance falls below the normal range.
    assert_eq!((compared, outside), (1581, 99));
}

/// `y → y + k` on Gaussian: the response's shift moves only the intercept
/// (the design has one), so the fit on `y + k` is the fit on `y` with the
/// intercept `+ k`: the log-likelihood, every other β, the SEs, every
/// `varcorr` entry and σ̂² unchanged. The reference is the fit on
/// `y' = (y + k) − k`, which is exact in f64 (Sterbenz), so the two fits see
/// the same data and only the algorithm can tell them apart.
#[test]
fn response_plus_k_keeps_the_gaussian_fit() {
    // Largest seen: 4.8e-15 on the log-likelihood, 1.05e-6 on the map (a
    // `varcorr` entry of the crossed ML fit at k = 1e5). y + k is not exact in
    // f64, so the two searches may stop at different points of one flat
    // optimum; the log-likelihood band is what shows it is the same optimum.
    const LOGLIK_TOL: f64 = 1e-8;
    let models = [
        ("gaussian OLS", ModelSpec::fixed(Family::Gaussian), false),
        (
            "gaussian (1 | g)",
            ModelSpec::mixed(Family::Gaussian, Re::intercept()),
            false,
        ),
        (
            "gaussian (1 | g) + (1 + x | h)",
            ModelSpec::mixed(
                Family::Gaussian,
                Re::intercept().extra(Grouping::crossed().slopes([1])),
            ),
            true,
        ),
    ];
    let x = design();
    let (g, h) = ids();
    let y0 = gaussian_y();
    let (mut cases, mut failures) = (0, Vec::new());
    for (name, spec, crossed) in &models {
        let mut gids = GroupIds::new(g.clone());
        if *crossed {
            gids = gids.extra(h.clone());
        }
        for reml in [false, true] {
            let mut opts = FitOptions::default();
            opts.reml = Some(reml);
            let fit_on = |y: &[f64]| {
                let mut data = Data::new(&x, y, N, P);
                if spec.re.is_some() {
                    data = data.ids(&gids);
                }
                fit(&data, spec, &opts).unwrap_or_else(|e| panic!("{name}: {e}"))
            };
            for k in [1e5, 1e8, 1e10] {
                let shifted: Vec<f64> = y0.iter().map(|v| v + k).collect();
                let back: Vec<f64> = shifted.iter().map(|v| v - k).collect();
                let base = fit_on(&back);
                assert!(base.converged(), "{name}: the base fit converges");
                assert!(!base.singular(), "{name}: the base fit is not singular");
                let f = fit_on(&shifted);
                let at = format!("{name} reml = {reml} at k = {k:e}");
                let mut check = |what: &str, got: f64, want: f64, tol: f64| {
                    let d = rel(got, want);
                    if d.is_nan() || d > tol {
                        failures.push(format!("{at}: {what} {got:e} vs {want:e} (rel {d:e})"));
                    }
                };
                check("loglik", f.loglik(), base.loglik(), LOGLIK_TOL);
                check("intercept", f.beta()[0], base.beta()[0] + k, TOL);
                check("slope", f.beta()[1], base.beta()[1], TOL);
                for j in 0..P {
                    check("se", f.se()[j], base.se()[j], TOL);
                }
                for (a, b) in every_varcorr(&f).iter().zip(every_varcorr(&base)) {
                    check("varcorr", *a, b, TOL);
                }
                check("dispersion", f.dispersion(), base.dispersion(), TOL);
                cases += 1;
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(cases, 18);
}

/// `y → c·y` on Gaussian: `y = Xβ + Zb + e` scales as a whole, so β, the SEs,
/// the random effects and the fitted values take `c`, `vcov`, every
/// `varcorr` entry, the conditional variances and σ̂² take `c²`, the
/// log-likelihood takes `− ν·ln c` and the LMM deviance `+ 2ν·ln c`, with
/// ν = n on ML and n − p on REML (whose criterion is the density of the
/// n − p error contrasts).
///
/// The map is checked entry by entry at every `c`. A mapped value outside
/// the normal range, and an SE whose variance is, has no exact value to
/// compare and is counted apart. At a `c` other than a power of two the
/// search may stop elsewhere on the same optimum, so the objective is
/// compared there too.
#[test]
fn response_times_c_maps_back_on_gaussian() {
    // Largest seen on the first passing run: 3.3e-16 relative on the map at
    // the powers of two (the log-likelihood; every other entry 0), 2.7e-11 at
    // 1e-200 and 1e80 (a `varcorr` entry), 1.5e-11 on the objective.
    let models = [
        ("gaussian OLS", ModelSpec::fixed(Family::Gaussian)),
        (
            "gaussian (1 | g)",
            ModelSpec::mixed(Family::Gaussian, Re::intercept()),
        ),
    ];
    let x = design();
    let (g, _) = ids();
    let gids = GroupIds::new(g);
    let y0 = gaussian_y();
    let (mut map_cases, mut objective_cases, mut compared, mut outside) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    for (name, spec) in &models {
        let mixed = spec.re.is_some();
        for reml in [false, true] {
            let mut opts = FitOptions::default();
            opts.reml = Some(reml);
            opts.condvar = true;
            let fit_at = |c: f64| {
                let y: Vec<f64> = y0.iter().map(|v| c * v).collect();
                let mut data = Data::new(&x, &y, N, P);
                if mixed {
                    data = data.ids(&gids);
                }
                fit(&data, spec, &opts)
                    .unwrap_or_else(|e| panic!("{name} reml = {reml} at c = {c:e}: {e}"))
            };
            let objective = |f: &Fit| {
                if f.deviance().is_finite() {
                    f.deviance()
                } else {
                    -2.0 * f.loglik()
                }
            };
            let nu = if reml { (N - P) as f64 } else { N as f64 };
            let base = fit_at(1.0);
            assert!(base.converged(), "{name}: the base fit converges");
            assert!(!base.singular(), "{name}: the base fit is not singular");
            for c in [
                2f64.powi(-800),
                2f64.powi(-664),
                2f64.powi(266),
                1e-200,
                1e80,
            ] {
                let f = fit_at(c);
                let at = format!("{name} reml = {reml} at c = {c:e}");
                if f.converged() != base.converged() {
                    failures.push(format!("{at}: converged flipped"));
                }
                // `want = factor·b` has an exact f64 value when it is normal or
                // `b` is 0.
                let mut check = |what: &str, got: f64, want: f64, b: f64, exact_range: bool| {
                    if !(exact_range && (want.is_normal() || b == 0.0)) {
                        outside += 1;
                        return;
                    }
                    compared += 1;
                    let d = rel(got, want);
                    if d.is_nan() || d > TOL {
                        failures.push(format!("{at}: {what} {got:e} vs {want:e} (rel {d:e})"));
                    }
                };
                for j in 0..P {
                    let b = base.beta()[j];
                    check("beta", f.beta()[j], c * b, b, true);
                    let (b, se) = (base.se()[j], c * base.se()[j]);
                    check("se", f.se()[j], se, b, (se * se).is_normal());
                }
                for (a, &b) in f.vcov().iter().zip(base.vcov()) {
                    check("vcov", *a, c * c * b, b, true);
                }
                if mixed {
                    for (a, b) in every_varcorr(&f).iter().zip(every_varcorr(&base)) {
                        check("varcorr", *a, c * c * b, b, true);
                    }
                    for (a, &b) in f.ranef(0).iter().zip(base.ranef(0)) {
                        check("ranef", *a, c * b, b, true);
                    }
                    for (a, &b) in f.condvar(0).iter().zip(base.condvar(0)) {
                        check("condvar", *a, c * c * b, b, true);
                    }
                    let b = base.deviance();
                    check("deviance", f.deviance(), b + 2.0 * nu * c.ln(), b, true);
                }
                let b = base.dispersion();
                check("dispersion", f.dispersion(), c * c * b, b, true);
                for (a, &b) in f.fitted().iter().zip(base.fitted()) {
                    check("fitted", *a, c * b, b, true);
                }
                let b = base.loglik();
                check("loglik", f.loglik(), b - nu * c.ln(), b, true);
                map_cases += 1;
            }
            for c in [1e-200, 1e80] {
                let f = fit_at(c);
                let at = format!("{name} reml = {reml} at c = {c:e}");
                if f.converged() != base.converged() {
                    failures.push(format!("{at}: converged flipped"));
                }
                let want = objective(&base) + 2.0 * nu * c.ln();
                let d = (objective(&f) - want).abs();
                if d.is_nan() || d > DEV_EPS {
                    failures.push(format!("{at}: objective moved by {d:e}"));
                }
                objective_cases += 1;
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!((map_cases, objective_cases), (20, 8));
    // Outside the normal range: at the three small scales every variance (SEs,
    // `vcov`, σ̂², and on the LMM `varcorr` and the conditional variances).
    assert_eq!((compared, outside), (1042, 138));
}

/// A log-link offset on Gamma and inverse Gaussian is a rescaled response:
/// both are scale families, so `log μᵢ = oᵢ + xᵢβ` on `y` is `log μᵢ' = xᵢβ`
/// on `yᵢ' = yᵢ·e^(−oᵢ)` with no offset. β is the same, the intercept
/// included; the Gamma dispersion is the same, and the inverse-Gaussian one
/// on `y` is the one on `y'` times `e^(−o)` (its unit deviance scales as
/// `1/c`);
/// the log-likelihood on `y` is the one on `y'` minus `Σ oᵢ`, the Jacobian
/// of the rescale. The offsets sit far from 0 (mean about 5), where the
/// fit centres them.
#[test]
fn a_log_link_offset_is_a_rescaled_response() {
    // Largest seen on the first passing run: 1.1e-7 relative (the Gamma
    // GLMM's dispersion; the GLMs 5e-14).
    let gamma_log = Family::Gamma {
        link: GammaLink::Log,
    };
    let ig_log = Family::InverseGaussian {
        link: InverseGaussianLink::Log,
    };
    let per_row: Vec<f64> = (0..N)
        .map(|i| 5.0 + 3.0 * (1.3 * i as f64 + 0.4).sin())
        .collect();
    let constant = vec![5.0; N];
    let models = [
        ("gamma log GLM", ModelSpec::fixed(gamma_log), &per_row),
        (
            "gamma log (1 | g) Laplace",
            ModelSpec::mixed(gamma_log, Re::intercept()),
            &per_row,
        ),
        (
            "inverse gaussian log GLM",
            ModelSpec::fixed(ig_log),
            &constant,
        ),
    ];
    let x = design();
    let (g, _) = ids();
    let gids = GroupIds::new(g);
    let y0 = positive_y();
    let opts = FitOptions::default();
    let (mut cases, mut failures) = (0, Vec::new());
    for (name, spec, offset) in &models {
        let fit_on = |y: &[f64], offset: Option<&[f64]>| {
            let mut data = Data::new(&x, y, N, P);
            if let Some(o) = offset {
                data = data.offset(o);
            }
            if spec.re.is_some() {
                data = data.ids(&gids);
            }
            fit(&data, spec, &opts).unwrap_or_else(|e| panic!("{name}: {e}"))
        };
        // The response carries its offset, so the base fit's data are near `y0`.
        let y: Vec<f64> = y0.iter().zip(*offset).map(|(v, o)| v * o.exp()).collect();
        let rescaled: Vec<f64> = y.iter().zip(*offset).map(|(v, o)| v * (-o).exp()).collect();
        let base = fit_on(&rescaled, None);
        assert!(base.converged(), "{name}: the base fit converges");
        assert!(!base.singular(), "{name}: the base fit is not singular");
        let f = fit_on(&y, Some(offset));
        if f.converged() != base.converged() {
            failures.push(format!("{name}: converged flipped"));
        }
        let mut check = |what: &str, got: f64, want: f64| {
            let d = rel(got, want);
            if d.is_nan() || d > TOL {
                failures.push(format!("{name}: {what} {got:e} vs {want:e} (rel {d:e})"));
            }
        };
        for j in 0..P {
            check("beta", f.beta()[j], base.beta()[j]);
        }
        let disp = if matches!(spec.family, Family::InverseGaussian { .. }) {
            base.dispersion() * (-offset[0]).exp()
        } else {
            base.dispersion()
        };
        check("dispersion", f.dispersion(), disp);
        let sum_o: f64 = offset.iter().sum();
        check("loglik", f.loglik(), base.loglik() - sum_o);
        cases += 1;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(cases, 3);
}

fn binomial_logit() -> Family {
    Family::Binomial {
        link: BinomialLink::Logit,
    }
}

/// Swapping the declaration order of two crossed groupings is the same model:
/// every per-grouping output trades places (grouping `d` of one spelling is
/// `1 − d` of the other) and every other output stays. Each pair's second
/// spelling is one the size rule reorders inside the kernel (`h`, the
/// smaller grouping, written first; the lone slope grouping written last), so
/// the pair compares the reordered path with the one that is not. On the
/// Gaussian pairs the profile intervals trade places too.
#[test]
fn swapping_two_crossed_groupings_permutes_every_output() {
    // Largest seen on the first run: 2.2e-16 relative (a Gaussian `coef_df`);
    // every other value, the profile rows included, equal bit for bit.
    let x = design();
    let (g, h) = ids();
    let gh = GroupIds::new(g.clone()).extra(h.clone());
    let hg = GroupIds::new(h).extra(g);
    let trials = vec![TRIALS; N];
    let shapes = [
        (
            "(1 | g) + (1 | h)",
            Re::intercept().extra(Grouping::crossed()),
            Re::intercept().extra(Grouping::crossed()),
        ),
        (
            "(1 | g) + (1 + x | h)",
            Re::intercept().extra(Grouping::crossed().slopes([1])),
            Re::intercept().slopes([1]).extra(Grouping::crossed()),
        ),
    ];
    let mut opts = FitOptions::default();
    opts.condvar = true;
    let mut t = Tally::default();
    let mut profile_pairs = 0;
    for family in [Family::Gaussian, binomial_logit()] {
        let gaussian = family == Family::Gaussian;
        let y = if gaussian {
            gaussian_y()
        } else {
            binomial_share()
        };
        let w = (!gaussian).then_some(trials.as_slice());
        for (shape, re_a, re_b) in &shapes {
            let name = format!("{family:?} {shape}");
            let (spec_a, spec_b) = (
                ModelSpec::mixed(family, re_a.clone()),
                ModelSpec::mixed(family, re_b.clone()),
            );
            let (data_a, data_b) = (data(&x, &y, w, Some(&gh)), data(&x, &y, w, Some(&hg)));
            let a = fit(&data_a, &spec_a, &opts).unwrap_or_else(|e| panic!("{name}: {e}"));
            let b = fit(&data_b, &spec_b, &opts).unwrap_or_else(|e| panic!("{name} swapped: {e}"));
            assert_clean_base(&a, &name);
            assert_eq!(a.ranef_levels(), [8, 5], "{name}: g declared first");
            assert_eq!(b.ranef_levels(), [5, 8], "{name}: h declared first");
            let mut c = t.case(name.clone());
            c.objective(objective(&b), objective(&a));
            c.close("loglik", b.loglik(), a.loglik());
            c.close_all("beta", b.beta(), a.beta());
            c.close_all("se", b.se(), a.se());
            c.close_all("coef_df", b.coef_df(), a.coef_df());
            c.close_all("fitted", b.fitted(), a.fitted());
            c.close("dispersion", b.dispersion(), a.dispersion());
            c.same("converged", b.converged(), a.converged());
            c.same("singular", b.singular(), a.singular());
            c.same(
                "boundary",
                b.diagnostics().boundary,
                a.diagnostics().boundary,
            );
            for d in 0..2 {
                let e = 1 - d;
                c.same("n_re", b.n_re(e), a.n_re(d));
                c.same("ranef_levels", b.ranef_levels()[e], a.ranef_levels()[d]);
                c.same(
                    "pinned",
                    &b.diagnostics().pinned[e],
                    &a.diagnostics().pinned[d],
                );
                c.close_all("varcorr", &b.varcorr(e), &a.varcorr(d));
                c.close_all("stddev_se", &b.stddev_se(e), &a.stddev_se(d));
                c.close_all("ranef", b.ranef(e), a.ranef(d));
                c.close_all("condvar", b.condvar(e), a.condvar(d));
            }
            if gaussian {
                let pa = a.profile_intervals(&data_a, &spec_a, 0.95, None);
                let pb = b.profile_intervals(&data_b, &spec_b, 0.95, None);
                match (pa, pb) {
                    (Ok(pa), Ok(pb)) => {
                        c.same("profile rows", pb.rows.len(), pa.rows.len());
                        for ra in &pa.rows {
                            let want = match ra.param {
                                ProfileParam::Sd { group, term } => ProfileParam::Sd {
                                    group: 1 - group,
                                    term,
                                },
                                ProfileParam::Cor { group, row, col } => ProfileParam::Cor {
                                    group: 1 - group,
                                    row,
                                    col,
                                },
                                other => other,
                            };
                            let Some(rb) = pb.rows.iter().find(|r| r.param == want) else {
                                c.same("profile row", None, Some(want));
                                continue;
                            };
                            let what = format!("profile {:?}", ra.param);
                            c.close(&format!("{what} estimate"), rb.estimate, ra.estimate);
                            c.close(&format!("{what} lower"), rb.lower, ra.lower);
                            c.close(&format!("{what} upper"), rb.upper, ra.upper);
                        }
                    }
                    (pa, pb) => c.fail(format!("profile intervals {pa:?} and {pb:?}")),
                }
                profile_pairs += 1;
            }
            c.end(&b);
        }
    }
    t.assert_clean();
    assert_eq!((t.cases, profile_pairs), (4, 2));
    assert_eq!((t.compared, t.flagged), (386, 0));
}

/// Relabelling a factor's levels, and declaring levels no row has, is the
/// same model: the fitted values, every `varcorr` entry, the level counts and
/// the objective stay. β is not compared, since reversing the levels moves
/// the treatment-contrast base and β is then another parametrization. Both
/// transforms reach the fixed factor and both grouping factors, and the
/// unused level sits first in the fixed factor, before its base.
#[cfg(feature = "formula")]
#[test]
fn relabelled_and_unused_levels_leave_the_fit_unchanged() {
    // Largest seen on the first run: 6.1e-10 relative (a Gaussian `varcorr`
    // entry, relabelled), the objective equal to 7.1e-15.
    let base_table = table();
    let models = [
        (
            "gaussian",
            "y ~ x + f + (1 | g) + (1 | h)",
            Family::Gaussian,
        ),
        (
            "binomial logit",
            "cbind(k, nk) ~ x + f + (1 | g) + (1 | h)",
            binomial_logit(),
        ),
    ];
    let opts = FitOptions::default();
    let mut t = Tally::default();
    for (name, formula, family) in models {
        let fit_on = |table: &glmm::formula::Table| {
            glmm::formula::fit(formula, table, family, &opts)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
        };
        let base = fit_on(&base_table);
        let base = base.fit();
        assert_clean_base(base, name);
        for (transform, unused, reverse) in [
            ("relabelled", false, true),
            ("unused levels", true, false),
            ("relabelled and unused levels", true, true),
        ] {
            let twin = fit_on(&relabelled(&base_table, unused, reverse));
            let f = twin.fit();
            let mut c = t.case(format!("{name}, {transform}"));
            c.objective(objective(f), objective(base));
            c.close_all("fitted", f.fitted(), base.fitted());
            for d in 0..base.n_groupings() {
                c.close_all("varcorr", &f.varcorr(d), &base.varcorr(d));
            }
            c.same("ranef_levels", f.ranef_levels(), base.ranef_levels());
            c.same("converged", f.converged(), base.converged());
            c.same("singular", f.singular(), base.singular());
            c.end(f);
        }
    }
    t.assert_clean();
    assert_eq!((t.cases, t.compared, t.flagged), (6, 252, 0));
}

/// The table: `y` (Gaussian), `k` and `nk` (binomial successes and
/// failures), `x`, and the factors `f` (3 levels by `i % 3`), `g` and `h`
/// (the two crossed groupings of `ids`).
#[cfg(feature = "formula")]
fn table() -> glmm::formula::Table {
    use glmm::formula::Column;
    let factor = |prefix: &str, l: u32, codes: Vec<u32>| Column::Factor {
        levels: (0..l).map(|j| format!("{prefix}{j}")).collect(),
        codes,
    };
    let (g, h) = ids();
    let k = binomial_k();
    glmm::formula::Table {
        columns: vec![
            ("y".into(), Column::Numeric(gaussian_y())),
            (
                "nk".into(),
                Column::Numeric(k.iter().map(|k| TRIALS - k).collect()),
            ),
            ("k".into(), Column::Numeric(k)),
            ("x".into(), Column::Numeric((0..N).map(x_seq).collect())),
            (
                "f".into(),
                Column::Factor {
                    levels: ["p", "q", "r"].map(String::from).to_vec(),
                    codes: (0..N as u32).map(|i| i % 3).collect(),
                },
            ),
            ("g".into(), factor("g", 8, g)),
            ("h".into(), factor("h", 5, h)),
        ],
        n: N,
    }
}

/// `t` with every factor recoded. `reverse`: the levels in reverse order
/// under new names (level `j` of `L` becomes `"L{L−1−j}"`, listed at
/// position `L−1−j`). `unused`: `"zzz"` declared first and `"yyy"` in the
/// middle, neither used by any row.
#[cfg(feature = "formula")]
fn relabelled(t: &glmm::formula::Table, unused: bool, reverse: bool) -> glmm::formula::Table {
    use glmm::formula::Column;
    let columns = t
        .columns
        .iter()
        .map(|(name, col)| {
            let col = match col {
                Column::Numeric(v) => Column::Numeric(v.clone()),
                Column::Factor { levels, codes } => {
                    let l = levels.len() as u32;
                    let (mut levels, mut codes) = (levels.clone(), codes.clone());
                    if reverse {
                        levels = (0..l).map(|j| format!("L{j}")).collect();
                        codes.iter_mut().for_each(|c| *c = l - 1 - *c);
                    }
                    if unused {
                        let mid = l / 2;
                        levels.insert(mid as usize, "yyy".into());
                        levels.insert(0, "zzz".into());
                        codes
                            .iter_mut()
                            .for_each(|c| *c += 1 + u32::from(*c >= mid));
                    }
                    Column::Factor { levels, codes }
                }
            };
            (name.clone(), col)
        })
        .collect();
    glmm::formula::Table { columns, n: t.n }
}

/// `v → T·v` for a 2 × 2 `T`.
fn map2(t: [[f64; 2]; 2], v: &[f64]) -> [f64; 2] {
    [
        t[0][0] * v[0] + t[0][1] * v[1],
        t[1][0] * v[0] + t[1][1] * v[1],
    ]
}

/// `M → T·M·Tᵀ` for 2 × 2 row-major `M`.
fn sandwich2(t: [[f64; 2]; 2], m: &[f64]) -> [f64; 4] {
    let tm = [map2(t, &[m[0], m[2]]), map2(t, &[m[1], m[3]])]; // columns of T·M
    let r0 = map2(t, &[tm[0][0], tm[1][0]]); // (T·M·Tᵀ) row 0 = T·(row 0 of T·M)
    let r1 = map2(t, &[tm[0][1], tm[1][1]]);
    [r0[0], r0[1], r1[0], r1[1]]
}

/// A predictor transform of the location-and-scale rows: `x × c` and
/// `x + a`, with `T` the map they put on (intercept, slope) and the shift
/// they put on the REML objective (`log|X′V⁻¹X|` moves by `2·ln|det T⁻¹|`).
/// `x + a` is compared with the fit on `(x + a) − a`, which is exact in f64
/// (Sterbenz), so both fits see the same `x` and only the algorithm can tell
/// them apart. `a` stops at 1e6: from 1e7 on, the alias gate drops `x + a`
/// as collinear with the intercept (as R's `lm` does at its 1e-7 tolerance).
struct Transform {
    name: String,
    x_base: Vec<f64>,
    x_twin: Vec<f64>,
    t: [[f64; 2]; 2],
    reml_shift: f64,
}

fn predictor_transforms() -> Vec<Transform> {
    let x0 = design();
    let mut out = Vec::new();
    for c in [1e-100, 1e-8, 1e8, 1e100] {
        let scaled = (0..N).flat_map(|i| [1.0, c * x_seq(i)]).collect();
        let t = [[1.0, 0.0], [0.0, 1.0 / c]];
        out.push(Transform {
            name: format!("x × {c:e}"),
            x_base: x0.clone(),
            x_twin: scaled,
            t,
            reml_shift: 2.0 * c.ln(),
        });
    }
    for a in [1e5, 1e6] {
        let shifted: Vec<f64> = (0..N).flat_map(|i| [1.0, x_seq(i) + a]).collect();
        let back = shifted.chunks(2).flat_map(|r| [1.0, r[1] - a]).collect();
        let t = [[1.0, -a], [0.0, 1.0]];
        out.push(Transform {
            name: format!("x + {a:e}"),
            x_base: back,
            x_twin: shifted,
            t,
            reml_shift: 0.0,
        });
    }
    out
}

/// `x → c·x` and `x → x + a` on column 1 of the design: with `T` the map on
/// (intercept, slope) (`diag(1, 1/c)`, or `[[1, −a], [0, 1]]`), β → T·β,
/// `vcov` → T·V·Tᵀ, the random slope grouping's `varcorr` → T·Σ·Tᵀ and each
/// of its levels' modes → T·b, the intercept-only grouping, the fitted
/// values and the dispersion unchanged, the ML objective unchanged and the
/// REML one `+ 2·ln c` under `x × c`.
#[test]
fn predictor_location_and_scale_map_the_fit() {
    // Largest seen on the first run: 5.0e-7 relative (an `h` `varcorr` entry
    // of the crossed REML fit at x × 1e100), the objective equal to 3.4e-13.
    let (g, h) = ids();
    let gids = GroupIds::new(g.clone());
    let ghids = GroupIds::new(g).extra(h);
    let models = [
        (
            "gaussian (1 | g)",
            ModelSpec::mixed(Family::Gaussian, Re::intercept()),
            Some(&gids),
        ),
        (
            "gaussian (1 | g) + (1 + x | h)",
            ModelSpec::mixed(
                Family::Gaussian,
                Re::intercept().extra(Grouping::crossed().slopes([1])),
            ),
            Some(&ghids),
        ),
        (
            "binomial logit GLM",
            ModelSpec::fixed(binomial_logit()),
            None,
        ),
    ];
    let trials = vec![TRIALS; N];
    let transforms = predictor_transforms();
    let mut t = Tally::default();
    for &(name, ref spec, ids) in &models {
        let gaussian = spec.family == Family::Gaussian;
        let (y, w) = if gaussian {
            (gaussian_y(), None)
        } else {
            (binomial_share(), Some(trials.as_slice()))
        };
        let remls: &[Option<bool>] = if gaussian {
            &[Some(false), Some(true)]
        } else {
            &[None]
        };
        for &reml in remls {
            let mut opts = FitOptions::default();
            opts.reml = reml;
            let fit_on = |x: &[f64]| {
                fit(&data(x, &y, w, ids), spec, &opts).unwrap_or_else(|e| panic!("{name}: {e}"))
            };
            for Transform {
                name: transform,
                x_base,
                x_twin,
                t: tm,
                reml_shift,
            } in &transforms
            {
                let base = fit_on(x_base);
                assert_clean_base(&base, name);
                let f = fit_on(x_twin);
                let mut c = t.case(format!("{name} reml = {reml:?}, {transform}"));
                let shift = if reml == Some(true) { *reml_shift } else { 0.0 };
                c.objective(objective(&f), objective(&base) + shift);
                let beta = map2(*tm, base.beta());
                let vcov = sandwich2(*tm, base.vcov());
                for j in 0..P {
                    c.mapped("beta", f.beta()[j], beta[j]);
                    c.mapped("se", f.se()[j], vcov[j * P + j].sqrt());
                }
                for (k, want) in vcov.iter().enumerate() {
                    c.mapped("vcov", f.vcov()[k], *want);
                }
                c.close_all("fitted", f.fitted(), base.fitted());
                c.close("dispersion", f.dispersion(), base.dispersion());
                c.same("converged", f.converged(), base.converged());
                c.same("singular", f.singular(), base.singular());
                if spec.re.is_some() {
                    c.close_all("varcorr g", &f.varcorr(0), &base.varcorr(0));
                    c.close_all("ranef g", f.ranef(0), base.ranef(0));
                }
                if base.n_groupings() == 2 {
                    let vc = sandwich2(*tm, &base.varcorr(1));
                    for (k, want) in vc.iter().enumerate() {
                        c.mapped("varcorr h", f.varcorr(1)[k], *want);
                    }
                    for (got, b) in f.ranef(1).chunks(2).zip(base.ranef(1).chunks(2)) {
                        let want = map2(*tm, b);
                        c.mapped("ranef h", got[0], want[0]);
                        c.mapped("ranef h", got[1], want[1]);
                    }
                }
                c.end(&f);
            }
        }
    }
    t.assert_clean();
    assert_eq!(
        (t.cases, t.compared, t.outside, t.flagged),
        (30, 1854, 0, 0)
    );
}

/// One case of the profile-interval rows: the ML profile intervals of `spec`
/// on `tr.x_base` and on `tr.x_twin`, mapped. Under `x × c` the intercept, σ,
/// the random intercepts' SDs and the correlation stay and the slope and the
/// random slope's SD take `1/c` (`c > 0`, so no bound swaps). Under `x + a`
/// the slope, σ, the random intercept SD of `g` and the random slope's SD
/// stay; the intercept, the random intercept SD of `h` and the correlation
/// have no closed-form map of their bounds and are not compared.
fn profile_case(t: &mut Tally, name: &str, spec: &ModelSpec, ids: &GroupIds, tr: &Transform) {
    let y = gaussian_y();
    let mut opts = FitOptions::default();
    opts.reml = Some(false);
    let profile = |x: &[f64]| {
        let d = data(x, &y, None, Some(ids));
        let f = fit(&d, spec, &opts).unwrap_or_else(|e| panic!("{name}: {e}"));
        let p = f.profile_intervals(&d, spec, 0.95, None);
        (f, p)
    };
    let (base, pb) = profile(&tr.x_base);
    assert_clean_base(&base, name);
    let (f, pf) = profile(&tr.x_twin);
    let mut c = t.case(format!("{name}, {}", tr.name));
    let scale = tr.t[0][1] == 0.0;
    match (pb, pf) {
        (Ok(pb), Ok(pf)) => {
            c.same("rows", pf.rows.len(), pb.rows.len());
            for rb in &pb.rows {
                // The factor the bounds take, or `None` where they have no
                // closed-form map.
                let k = match rb.param {
                    ProfileParam::Beta(1) | ProfileParam::Sd { group: 1, term: 1 } => {
                        Some(if scale { tr.t[1][1] } else { 1.0 })
                    }
                    ProfileParam::Beta(0)
                    | ProfileParam::Sd { group: 1, term: 0 }
                    | ProfileParam::Cor { .. } => scale.then_some(1.0),
                    _ => Some(1.0),
                };
                let Some(k) = k else { continue };
                let Some(rf) = pf.rows.iter().find(|r| r.param == rb.param) else {
                    c.same("row", None, Some(rb.param));
                    continue;
                };
                let what = format!("{:?}", rb.param);
                c.mapped(&format!("{what} estimate"), rf.estimate, k * rb.estimate);
                c.mapped(&format!("{what} lower"), rf.lower, k * rb.lower);
                c.mapped(&format!("{what} upper"), rf.upper, k * rb.upper);
            }
        }
        (pb, pf) => c.fail(format!("profile intervals {pb:?} and {pf:?}")),
    }
    c.end(&f);
}

fn gaussian_crossed_slope() -> ModelSpec {
    ModelSpec::mixed(
        Family::Gaussian,
        Re::intercept().extra(Grouping::crossed().slopes([1])),
    )
}

/// The profile intervals under the transforms of
/// `predictor_location_and_scale_map_the_fit`, on the Gaussian models by ML
/// ([`profile_case`]): `(1 | g)` under every transform, `(1 | g) + (1 + x | h)`
/// under `x × c`.
#[test]
fn predictor_location_and_scale_map_the_profile_intervals() {
    // Largest seen on the first run: 2.5e-9 relative (an SD estimate of the
    // crossed slope model at x × 1e100); every bound 3.6e-11 or less.
    let (g, h) = ids();
    let gids = GroupIds::new(g.clone());
    let ghids = GroupIds::new(g).extra(h);
    let intercept = ModelSpec::mixed(Family::Gaussian, Re::intercept());
    let slope = gaussian_crossed_slope();
    let mut t = Tally::default();
    for tr in &predictor_transforms() {
        profile_case(&mut t, "gaussian (1 | g)", &intercept, &gids, tr);
        if tr.t[0][1] == 0.0 {
            profile_case(&mut t, "gaussian (1 | g) + (1 + x | h)", &slope, &ghids, tr);
        }
    }
    t.assert_clean();
    assert_eq!((t.cases, t.compared, t.outside, t.flagged), (10, 150, 0, 0));
}

/// [`profile_case`] on `(1 | g) + (1 + x | h)` under `x + a`.
#[test]
#[ignore = "the profile search of a correlated random slope runs in the caller's coordinates, where shifting the slope's predictor drives the correlation towards -1 and the search cannot resolve the bounds"]
fn predictor_location_maps_the_profile_intervals_of_a_crossed_random_slope() {
    let (g, h) = ids();
    let ghids = GroupIds::new(g).extra(h);
    let slope = gaussian_crossed_slope();
    let mut t = Tally::default();
    for tr in predictor_transforms().iter().filter(|tr| tr.t[0][1] != 0.0) {
        profile_case(&mut t, "gaussian (1 | g) + (1 + x | h)", &slope, &ghids, tr);
    }
    t.assert_clean();
    assert_eq!((t.cases, t.compared, t.outside, t.flagged), (2, 24, 0, 0));
}

/// Every prior weight × `k` on a binomial and a Poisson GLM: the weights
/// carry no information about β beyond their ratios, so β stays, the SEs take
/// `1/√k` and the log-likelihood takes `k`. On Poisson the saturated term
/// carries the weight; on binomial it vanishes because the response is 0/1
/// (each row has 0 successes or 0 failures), so `−2·loglik` scales by `k`
/// exactly on both. The weights fall to `1/MAX_DATA_ABS`, the edge of their
/// range.
#[test]
fn tiny_prior_weights_rescale_the_binomial_and_poisson_glm() {
    // Largest seen on the first run: 6.7e-16 relative (a Poisson β at
    // k = 1e-12), the rescaled objective equal to 1.7e-13.
    let models = [
        ("binomial logit GLM", binomial_logit(), binary_y()),
        (
            "poisson log GLM",
            Family::Poisson {
                link: PoissonLink::Log,
            },
            count_y(),
        ),
    ];
    let x = design();
    let mut t = Tally::default();
    for (name, family, y) in &models {
        let spec = ModelSpec::fixed(*family);
        let fit_at = |k: f64| {
            let w = vec![k; N];
            fit(&data(&x, y, Some(&w), None), &spec, &FitOptions::default())
                .unwrap_or_else(|e| panic!("{name} at k = {k:e}: {e}"))
        };
        let base = fit_at(1.0);
        assert_clean_base(&base, name);
        for k in [1e-12, 1e-50, 1e-100] {
            let f = fit_at(k);
            let mut c = t.case(format!("{name} at k = {k:e}"));
            c.close_all("beta", f.beta(), base.beta());
            for j in 0..P {
                c.close("se", f.se()[j], base.se()[j] / k.sqrt());
            }
            c.objective(objective(&f) / k, objective(&base));
            c.same("converged", f.converged(), base.converged());
            c.end(&f);
        }
    }
    t.assert_clean();
    assert_eq!((t.cases, t.compared, t.flagged), (6, 24, 0));
}

/// On a GLMM, weights × `k` is not an invariance: the random-effect prior does
/// not scale with the weights, so at tiny weights the data carry almost no
/// information against it. The answer there is a zero variance: the fit ends
/// singular with every `varcorr` entry 0 and β at the unit-weight GLM's, or
/// it carries a note.
#[test]
#[ignore = "binomial and Poisson GLMMs whose prior weights are all far below 1 stop at their start and report it as converged"]
fn tiny_prior_weights_give_the_binomial_and_poisson_glmm_a_zero_variance() {
    let models = [
        ("binomial logit (1 | g)", binomial_logit(), binary_y()),
        (
            "poisson log (1 | g)",
            Family::Poisson {
                link: PoissonLink::Log,
            },
            count_y(),
        ),
    ];
    let x = design();
    let gids = GroupIds::new(ids().0);
    let opts = FitOptions::default();
    let mut t = Tally::default();
    for (name, family, y) in &models {
        let ones = vec![1.0; N];
        let glm = fit(
            &data(&x, y, Some(&ones), None),
            &ModelSpec::fixed(*family),
            &opts,
        )
        .unwrap_or_else(|e| panic!("{name} GLM: {e}"));
        assert!(glm.converged(), "{name}: the GLM converges");
        let spec = ModelSpec::mixed(*family, Re::intercept());
        for k in [1e-12, 1e-50, 1e-100] {
            let w = vec![k; N];
            let f = fit(&data(&x, y, Some(&w), Some(&gids)), &spec, &opts)
                .unwrap_or_else(|e| panic!("{name} at k = {k:e}: {e}"));
            let mut c = t.case(format!("{name} at k = {k:e}"));
            c.same("singular", f.singular(), true);
            for v in f.varcorr(0) {
                c.same("varcorr", v, 0.0);
            }
            c.close_all("beta", f.beta(), glm.beta());
            c.end(&f);
        }
    }
    t.assert_clean();
    assert_eq!(t.cases, 6);
}

/// No weights and explicit unit weights are the same fit, on every family.
/// On Gaussian the default options resolve all-equal weights to the weights
/// as given (no power search), so the pair also checks that resolution. The
/// binomial response is the proportion `k/20` with one trial per row.
#[test]
fn unit_weights_give_the_unweighted_fit_on_every_family() {
    // Largest seen on the first run: 3.4e-7 relative (the Gamma log GLMM's
    // `varcorr`), the objective equal to 1.1e-12.
    let families = [
        ("gaussian", Family::Gaussian, gaussian_y()),
        ("binomial logit", binomial_logit(), binomial_share()),
        (
            "poisson log",
            Family::Poisson {
                link: PoissonLink::Log,
            },
            count_y(),
        ),
        (
            "negative binomial log",
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            count_y(),
        ),
        (
            "gamma log",
            Family::Gamma {
                link: GammaLink::Log,
            },
            positive_y(),
        ),
        (
            "inverse gaussian log",
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
            positive_y(),
        ),
    ];
    let x = design();
    let gids = GroupIds::new(ids().0);
    let ones = vec![1.0; N];
    let opts = FitOptions::default();
    let mut t = Tally::default();
    for (fname, family, y) in &families {
        let mut specs = vec![("fixed", ModelSpec::fixed(*family))];
        // The crate fits inverse Gaussian as a GLM only.
        if !matches!(family, Family::InverseGaussian { .. }) {
            specs.push(("(1 | g)", ModelSpec::mixed(*family, Re::intercept())));
        }
        for (shape, spec) in specs {
            let name = format!("{fname} {shape}");
            let ids = spec.re.as_ref().map(|_| &gids);
            let fit_with = |w: Option<&[f64]>| {
                fit(&data(&x, y, w, ids), &spec, &opts).unwrap_or_else(|e| panic!("{name}: {e}"))
            };
            let base = fit_with(None);
            assert_clean_base(&base, &name);
            let f = fit_with(Some(&ones));
            let mut c = t.case(name.clone());
            c.close_all("beta", f.beta(), base.beta());
            c.close_all("se", f.se(), base.se());
            for d in 0..base.n_groupings() {
                c.close_all("varcorr", &f.varcorr(d), &base.varcorr(d));
            }
            c.close("dispersion", f.dispersion(), base.dispersion());
            c.close_all("fitted", f.fitted(), base.fitted());
            c.objective(-2.0 * f.loglik(), -2.0 * base.loglik());
            c.objective(objective(&f), objective(&base));
            c.same("converged", f.converged(), base.converged());
            c.same("singular", f.singular(), base.singular());
            c.end(&f);
        }
    }
    t.assert_clean();
    assert_eq!((t.cases, t.compared, t.flagged), (11, 500, 0));
}

/// The lower Cholesky factor of a `q × q` row-major covariance, as the
/// column-major vech `θ` takes.
fn chol_vech(s: &[f64], q: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; q * q];
    for c in 0..q {
        let d = s[c * q + c] - (0..c).map(|k| l[c * q + k].powi(2)).sum::<f64>();
        l[c * q + c] = d.max(0.0).sqrt();
        for r in c + 1..q {
            let off = s[r * q + c] - (0..c).map(|k| l[r * q + k] * l[c * q + k]).sum::<f64>();
            // A zero SD leaves its column of the factor zero.
            l[r * q + c] = if l[c * q + c] > 0.0 {
                off / l[c * q + c]
            } else {
                0.0
            };
        }
    }
    (0..q)
        .flat_map(|c| (c..q).map(move |r| (r, c)))
        .map(|(r, c)| l[r * q + c])
        .collect()
}

/// A warm start shortens the path to the cold fit's optimum and nothing
/// else: from far β starts, θ starts across six decades, the cold fit's own
/// estimates on the caller's scale (σ̂ divided out on Gaussian), and, on the
/// AGQ fits, the Laplace estimates. The cold objective is never above a warm
/// one by more than `DEV_EPS`, flagged or not (a warm fit that ends lower
/// means the cold fit missed the optimum); a warm fit either ends at the cold
/// optimum or carries a note. The slope models carry the slope column at
/// `100·x`, so the fit's internal column scale is not 1 and a θ on the
/// caller's scale differs from one on the internal scale.
#[test]
fn warm_starts_end_at_the_cold_optimum_or_carry_a_note() {
    // Largest seen on the first run: 7.6e-6 relative (a `varcorr` entry of
    // the Poisson slope model from θ = 1e-3), the objective equal to 3.2e-11.
    let families = [
        ("gaussian", Family::Gaussian),
        ("binomial logit", binomial_logit()),
        (
            "poisson log",
            Family::Poisson {
                link: PoissonLink::Log,
            },
        ),
    ];
    let x = design();
    let x100: Vec<f64> = (0..N).flat_map(|i| [1.0, 100.0 * x_seq(i)]).collect();
    let (g, h) = ids();
    let gids = GroupIds::new(g.clone());
    let ghids = GroupIds::new(g).extra(h);
    let trials = vec![TRIALS; N];
    let mut t = Tally::default();
    for (fname, family) in families {
        let gaussian = family == Family::Gaussian;
        let (y, w) = match family {
            Family::Gaussian => (gaussian_y(), None),
            Family::Binomial { .. } => (binomial_share(), Some(trials.as_slice())),
            _ => (count_y(), None),
        };
        let shapes = [
            ("(1 | g)", Re::intercept(), &x, &gids, false),
            (
                "(1 | g) + (1 + x | h)",
                Re::intercept().extra(Grouping::crossed().slopes([1])),
                &x100,
                &ghids,
                true,
            ),
        ];
        for (shape, re, x, ids, slope) in shapes {
            let name = format!("{fname} {shape}");
            let spec = ModelSpec::mixed(family, re);
            let d = data(x, &y, w, Some(ids));
            let opts = FitOptions::default();
            let cold = fit(&d, &spec, &opts).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_clean_base(&cold, &name);
            let s = |v: f64| if slope { vec![v, v, 0.0, v] } else { vec![v] };
            let mut starts: Vec<(String, StartValues, &Fit, u8)> = Vec::new();
            for v in [-28.0, 200.0, 1e14] {
                let start = StartValues::new(vec![v, 0.0], vec![]);
                starts.push((format!("beta_0 = {v:e}"), start, &cold, 1));
            }
            for v in [1e-3, 1e-1, 1e1, 1e3] {
                starts.push((
                    format!("theta = {v:e}"),
                    StartValues::new(vec![], s(v)),
                    &cold,
                    1,
                ));
            }
            if slope {
                let scale = if gaussian { cold.dispersion() } else { 1.0 };
                let theta = (0..2)
                    .flat_map(|d| {
                        let vc: Vec<f64> = cold.varcorr(d).iter().map(|v| v / scale).collect();
                        chol_vech(&vc, cold.n_re(d))
                    })
                    .collect();
                let start = StartValues::new(cold.beta().to_vec(), theta);
                starts.push(("the cold estimates".into(), start, &cold, 1));
            }
            let mut opts7 = FitOptions::default();
            opts7.nagq = 7;
            let cold7 = (!gaussian && !slope)
                .then(|| fit(&d, &spec, &opts7).unwrap_or_else(|e| panic!("{name} nagq 7: {e}")));
            if let Some(cold7) = &cold7 {
                assert_clean_base(cold7, &name);
                let theta = vec![cold.varcorr(0)[0].sqrt()];
                let start = StartValues::new(cold.beta().to_vec(), theta);
                starts.push(("nagq 7 from Laplace".into(), start, cold7, 7));
            }
            for (label, start, reference, nagq) in &starts {
                let mut c = t.case(format!("{name}, {label}"));
                let o = if *nagq == 7 { &opts7 } else { &opts };
                let warm = match fit_with_start(&d, &spec, start, o) {
                    Ok(f) => f,
                    Err(e) => {
                        c.fail(format!("{e}"));
                        continue;
                    }
                };
                let (oc, ow) = (objective(reference), objective(&warm));
                if oc.is_nan() || ow.is_nan() || oc > ow + DEV_EPS {
                    c.fail(format!("cold objective {oc:e} above warm {ow:e}"));
                }
                c.objective(ow, oc);
                c.close_all("beta", warm.beta(), reference.beta());
                c.close_all("se", warm.se(), reference.se());
                for d in 0..reference.n_groupings() {
                    c.close_all("varcorr", &warm.varcorr(d), &reference.varcorr(d));
                }
                c.same("converged", warm.converged(), reference.converged());
                c.end(&warm);
            }
        }
    }
    t.assert_clean();
    assert_eq!((t.cases, t.compared, t.flagged), (47, 331, 0));
}
