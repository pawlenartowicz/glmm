//! `Simulator`: reproducible draws, the buffer check, binomial trials from the
//! lowered weights, and the refused inputs.

use super::*;
use crate::formula::Column;
use crate::predict::predict_tests::{binomial, lowered_fit, BERN, SLEEP};
use crate::test_support::csv_table;
use crate::{FitError, PoissonLink};

const CBPP: &str = include_str!("../validation/data/empirical/cbpp.csv");

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// FNV-1a over the bit patterns, to compare draws across targets by eye.
fn hash(v: &[f64]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits().to_le_bytes().iter().fold(h, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

/// cbpp with `failures = size − incidence`, fitted as
/// `cbind(incidence, failures) ~ period + (1 | herd)`.
fn cbpp_fit() -> (crate::formula::Lowered, Fit) {
    let mut table = csv_table(CBPP, &["herd", "period"]);
    let numeric = |name: &str| match &table.columns.iter().find(|(n, _)| n == name).unwrap().1 {
        Column::Numeric(v) => v.clone(),
        Column::Factor { .. } => unreachable!("{name} is numeric"),
    };
    let failures: Vec<f64> = numeric("size")
        .iter()
        .zip(numeric("incidence"))
        .map(|(s, i)| s - i)
        .collect();
    table
        .columns
        .push(("failures".to_string(), Column::Numeric(failures)));
    let lo = crate::formula::lower(
        "cbind(incidence, failures) ~ period + (1 | herd)",
        &table,
        binomial(),
    )
    .unwrap();
    let fit = crate::fit(&lo.data(), lo.spec(), &crate::FitOptions::default()).unwrap();
    assert!(fit.converged());
    (lo, fit)
}

/// Each shape twice at one `(seed, draw)` is bit-equal, `simulate` equals
/// `simulate_into`, and another draw or seed differs. Prints a hash of the
/// parameters and of draw 0 so a wasm32 run can be compared with a native
/// one.
#[test]
fn the_same_seed_and_draw_give_the_same_response() {
    let shapes: [(&str, &[&str], &str, Family); 3] = [
        (
            SLEEP,
            &["Subject"],
            "Reaction ~ Days + (1 + Days | Subject)",
            Family::Gaussian,
        ),
        (
            BERN,
            &["subj", "g", "h"],
            "y ~ x + g + (1 | subj)",
            binomial(),
        ),
        (
            include_str!("../validation/data/simulated/sim_poisson_offset.csv"),
            &["cluster"],
            "y ~ x + offset(log_exposure) + (1 | cluster)",
            Family::Poisson {
                link: PoissonLink::Log,
            },
        ),
    ];
    let mut ran = 0;
    for (csv, factors, formula, family) in shapes {
        let (lo, fit) = lowered_fit(csv, factors, formula, family);
        let sim = Simulator::new(&fit, lo.spec(), &lo.design()).unwrap();
        let y = sim.simulate(0, 0);
        assert_eq!(y.len(), lo.n, "{formula}");
        assert!(y.iter().all(|v| v.is_finite()), "{formula}");
        assert_eq!(bits(&y), bits(&sim.simulate(0, 0)), "{formula}: twice");
        let mut into = vec![0.0; lo.n];
        sim.simulate_into(0, 0, &mut into).unwrap();
        assert_eq!(bits(&y), bits(&into), "{formula}: simulate_into");
        assert_ne!(bits(&y), bits(&sim.simulate(0, 1)), "{formula}: draw 1");
        assert_ne!(bits(&y), bits(&sim.simulate(1, 0)), "{formula}: seed 1");
        // The fit itself can differ by a few ULP between targets, so the
        // printed draw is taken at its parameters rounded to 7 digits, which
        // both targets parse to the same doubles.
        let round = |v: &f64| -> f64 { format!("{v:.6e}").parse().unwrap() };
        let beta: Vec<f64> = fit.beta.iter().map(round).collect();
        let varcorr: Vec<Vec<f64>> = fit
            .varcorr
            .iter()
            .map(|b| b.iter().map(round).collect())
            .collect();
        let fixed = FitParts {
            beta: &beta,
            varcorr: &varcorr,
            dispersion: round(&fit.dispersion),
            ..FitParts::of(&fit, lo.spec())
        };
        let parts: Vec<f64> = beta
            .iter()
            .chain(varcorr.iter().flatten())
            .chain([&fixed.dispersion])
            .copied()
            .collect();
        let y0 = Simulator::from_parts(&fixed, lo.spec(), &lo.design())
            .unwrap()
            .simulate(0, 0);
        println!(
            "{formula}: parameters {:016x}, draw 0 {:016x}",
            hash(&parts),
            hash(&y0)
        );
        ran += 1;
    }
    assert_eq!(ran, 3);
}

#[test]
fn simulate_into_checks_the_buffer() {
    let (lo, fit) = lowered_fit(
        SLEEP,
        &["Subject"],
        "Reaction ~ Days + (1 + Days | Subject)",
        Family::Gaussian,
    );
    let sim = Simulator::new(&fit, lo.spec(), &lo.design()).unwrap();
    for len in [lo.n - 1, lo.n + 1, 0] {
        let mut y = vec![0.0; len];
        assert!(
            matches!(sim.simulate_into(0, 0, &mut y), Err(FitError::Shape(_))),
            "length {len}"
        );
    }
    let mut y = vec![0.0; lo.n];
    assert!(sim.simulate_into(0, 0, &mut y).is_ok());
}

/// One effect is drawn per level a row carries, in id order, so ids spread up
/// to `u32::MAX` draw the same response as the dense ones, where sizing by
/// `max(id) + 1` would ask for about 4e9 draws and abort.
#[test]
fn sparse_ids_draw_like_dense_ones() {
    let (lo, fit) = lowered_fit(
        SLEEP,
        &["Subject"],
        "Reaction ~ Days + (1 + Days | Subject)",
        Family::Gaussian,
    );
    let dense = lo.design();
    let top = *lo.ids.primary.iter().max().unwrap();
    let spread = crate::GroupIds::new(
        (lo.ids.primary.iter())
            .map(|&id| if id == top { u32::MAX } else { id * 1000 })
            .collect(),
    );
    let sparse = dense.ids(&spread);
    let y = Simulator::new(&fit, lo.spec(), &dense)
        .unwrap()
        .simulate(3, 0);
    let y_sparse = Simulator::new(&fit, lo.spec(), &sparse)
        .unwrap()
        .simulate(3, 0);
    assert_eq!(bits(&y), bits(&y_sparse));
}

/// A `cbind` response simulates counts of the lowered trials: every `y·w` is a
/// whole number in `[0, w]`, and some lie strictly inside, which a 0/1 draw
/// would never give.
#[test]
fn simulated_counts_use_the_lowered_trials() {
    let (lo, fit) = cbpp_fit();
    let w = lo
        .weights
        .as_ref()
        .expect("cbind lowers the trials as weights");
    assert!(w.iter().any(|&w| w > 1.0));
    let sim = Simulator::new(&fit, lo.spec(), &lo.design()).unwrap();
    let mut inside = 0;
    for draw in 0..20 {
        for (y, &w) in sim.simulate(3, draw).iter().zip(w) {
            let k = y * w;
            assert!((k - k.round()).abs() <= 1e-9 * w, "draw {draw}: {k}");
            assert!((0.0..=w).contains(&k.round()), "draw {draw}: {k} of {w}");
            if k.round() > 0.0 && k.round() < w {
                inside += 1;
            }
        }
    }
    assert!(inside > 0);
}

#[test]
fn a_non_integer_trial_count_is_refused() {
    let (lo, fit) = cbpp_fit();
    let mut w = lo.weights.clone().unwrap();
    w[3] = 2.5;
    let design = lo.design().weights(&w);
    match Simulator::new(&fit, lo.spec(), &design) {
        Err(FitError::Weights(m)) => assert!(m.contains("row 3"), "{m}"),
        other => panic!("accepted or another error: {:?}", other.err()),
    }
}

#[test]
fn a_fit_without_a_dispersion_is_refused() {
    let (lo, mut fit) = lowered_fit(
        SLEEP,
        &["Subject"],
        "Reaction ~ Days + (1 + Days | Subject)",
        Family::Gaussian,
    );
    fit.dispersion = f64::NAN;
    assert!(matches!(
        Simulator::new(&fit, lo.spec(), &lo.design()),
        Err(FitError::Unsupported(_))
    ));
}

/// The Monte Carlo check of the draws against the fitted model's moments.
#[cfg(feature = "montecarlo-tests")]
mod montecarlo {
    use super::*;
    use crate::consts::{GH_NODES, GH_OFFSETS, GH_WEIGHTS};
    use crate::family::variance;
    use crate::{GammaLink, InverseGaussianLink, NegBinomialLink};
    use std::f64::consts::{PI, SQRT_2};

    /// Draws per model.
    const DRAWS: u64 = 4000;
    /// Standard errors a mean may sit from its model value: the Monte Carlo
    /// band, as `rlrt_refs::bands::RLRT_LEVEL_SE`. Set a priori.
    const MC_SE: f64 = 3.0;

    /// Per row of the design, the model's mean and variance of the response:
    /// over `η ~ N(xβ̂ + o, zᵀD̂z)`, `D̂` each grouping's `varcorr` block,
    /// `E[μ(η)]` and `E[φV(μ)/w] + Var[μ(η)]` by the 25-node Gauss-Hermite
    /// rule (exact on the Gaussian, whose `μ` is linear). `φ` is `σ̂²` or the
    /// Gamma and inverse-Gaussian `φ̂`, 1 on the other families, and `w` the
    /// prior weight where the family reads one (binomial trials, the
    /// Gaussian, Gamma and inverse-Gaussian precision).
    fn model_moments(fit: &Fit, spec: &ModelSpec, design: &Design<'_>) -> (Vec<f64>, Vec<f64>) {
        let population = PredictOptions {
            population: true,
            ..PredictOptions::default()
        };
        let eta = crate::predict(fit, spec, design, &population).unwrap().fit;
        let n = design.n;
        let mut tau2 = vec![0.0f64; n];
        if let (Some(re), Some(ids)) = (spec.re.as_ref(), design.ids) {
            let xm = MatRef::from_row_major_slice(design.x, n, design.p);
            let zm = MatRef::from_row_major_slice(design.re.unwrap_or(&[]), n, design.s);
            for ((_, cols), vech) in reported_columns(re, ids).into_iter().zip(&fit.varcorr) {
                let q = cols.len();
                for (i, t) in tau2.iter_mut().enumerate() {
                    let z: Vec<f64> = cols
                        .iter()
                        .map(|c| c.map_or(1.0, |c| slope_value(xm, zm, design.p, i, c)))
                        .collect();
                    let mut k = 0;
                    for c in 0..q {
                        for r in c..q {
                            let both = if r == c { 1.0 } else { 2.0 };
                            *t += both * vech[k] * z[r] * z[c];
                            k += 1;
                        }
                    }
                }
            }
        }
        let family = spec.family;
        let (phi, reads_weights) = match family {
            Family::Gaussian | Family::Gamma { .. } | Family::InverseGaussian { .. } => {
                (fit.dispersion, true)
            }
            Family::Binomial { .. } => (1.0, true),
            Family::Poisson { .. } | Family::NegativeBinomial { .. } => (1.0, false),
        };
        // Order 25 is block 12 of the table.
        let block = GH_OFFSETS[12]..GH_OFFSETS[13];
        let rule: Vec<(f64, f64)> = GH_NODES[block.clone()]
            .iter()
            .zip(&GH_WEIGHTS[block])
            .map(|(&x, &w)| (x, w / PI.sqrt()))
            .collect();
        let (mut mean, mut var) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let w = match (reads_weights, design.weights) {
                (true, Some(w)) => w[i],
                _ => 1.0,
            };
            let (mut m1, mut m2, mut within) = (0.0, 0.0, 0.0);
            for &(x, wk) in &rule {
                let mu = link_inv(family, eta[i] + SQRT_2 * tau2[i].sqrt() * x);
                m1 += wk * mu;
                m2 += wk * mu * mu;
                within += wk * phi * variance(family, fit.dispersion, mu) / w;
            }
            mean.push(m1);
            var.push(within + m2 - m1 * m1);
        }
        (mean, var)
    }

    /// The ten models of the check, fitted on their training designs: OLS,
    /// the sleepstudy LMM with a correlated, an uncorrelated and a
    /// random-effect-only slope block, and the GLMMs and GLM of every
    /// non-Gaussian family (binomial 0/1 and `cbind`, Poisson with an offset,
    /// Gamma, negative binomial, inverse Gaussian).
    fn ten_models() -> Vec<(crate::formula::Lowered, Fit, &'static str)> {
        let shapes: [(&str, &[&str], &str, Family); 9] = [
            (
                include_str!("../validation/data/simulated/sim_factorial.csv"),
                &["subj", "g", "h"],
                "y ~ g * h + x",
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
                BERN,
                &["subj", "g", "h"],
                "y ~ g * h + x + (1 | subj)",
                binomial(),
            ),
            (
                include_str!("../validation/data/simulated/sim_poisson_offset.csv"),
                &["cluster"],
                "y ~ 1 + x + offset(log_exposure) + (1 | cluster)",
                Family::Poisson {
                    link: PoissonLink::Log,
                },
            ),
            (
                include_str!("../validation/data/simulated/sim_gamma.csv"),
                &["cluster", "grp"],
                "y ~ 1 + x + grp + (1 | cluster)",
                Family::Gamma {
                    link: GammaLink::Log,
                },
            ),
            (
                include_str!("../validation/data/simulated/sim_nb.csv"),
                &["cluster", "grp"],
                "y ~ 1 + x + grp + (1 | cluster)",
                Family::NegativeBinomial {
                    link: NegBinomialLink::Log,
                },
            ),
            (
                include_str!("../validation/data/simulated/sim_igauss.csv"),
                &["grp"],
                "y ~ 1 + x + grp",
                Family::InverseGaussian {
                    link: InverseGaussianLink::Log,
                },
            ),
        ];
        let mut out: Vec<_> = shapes
            .into_iter()
            .map(|(csv, factors, formula, family)| {
                let (lo, fit) = lowered_fit(csv, factors, formula, family);
                (lo, fit, formula)
            })
            .collect();
        let (lo, fit) = cbpp_fit();
        out.insert(
            5,
            (lo, fit, "cbind(incidence, failures) ~ period + (1 | herd)"),
        );
        out
    }

    /// `DRAWS` draws of each model on its training design against the model's
    /// per-row moments. Rows that share a level are correlated, so each check
    /// is on a per-draw sum, with the standard error of its mean over the
    /// independent draws: `S_d = Σ_i (y_di − m_i)`, mean 0, and
    /// `T_d = Σ_i (y_di − m_i)²`, mean `Σ_i v_i`; each mean within
    /// [`MC_SE`] standard errors. A simulator that drew no random effects
    /// would leave `T_d` short by `Σ_i zᵢᵀD̂zᵢ` on every mixed model.
    #[test]
    fn simulated_moments_match_the_fitted_model() {
        let mut ran = 0;
        for (lo, fit, formula) in ten_models() {
            let design = lo.design();
            let (m, v) = model_moments(&fit, lo.spec(), &design);
            let sim = Simulator::new(&fit, lo.spec(), &design).unwrap();
            let r = DRAWS as f64;
            let (mut s1, mut s2, mut t1, mut t2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut y = vec![0.0; lo.n];
            for d in 0..DRAWS {
                sim.simulate_into(35, d, &mut y).unwrap();
                let (mut s, mut t) = (0.0, 0.0);
                for i in 0..lo.n {
                    let e = y[i] - m[i];
                    s += e;
                    t += e * e;
                }
                s1 += s;
                s2 += s * s;
                t1 += t;
                t2 += t * t;
            }
            let se = |sum: f64, sq: f64| ((sq - sum * sum / r) / (r - 1.0) / r).sqrt();
            let z_mean = (s1 / r) / se(s1, s2);
            let v_sum: f64 = v.iter().sum();
            let z_var = (t1 / r - v_sum) / se(t1, t2);
            let model = format!("{formula}, {:?}", lo.spec().family);
            println!("{model}: mean z {z_mean:.3}, variance z {z_var:.3}");
            assert!(z_mean.abs() <= MC_SE, "{model}: mean z {z_mean}");
            assert!(z_var.abs() <= MC_SE, "{model}: variance z {z_var}");
            ran += 1;
        }
        assert_eq!(ran, 10);
    }
}
