//! `Fit::r2`, `Fit::icc` and `variance_parts` on the shapes no frozen
//! reference covers. Each expected value is the formula computed here from the
//! fit's own fields and the raw data columns.

use std::f64::consts::PI;

use super::*;
use crate::formula::{self, Column, Lowered, Table};
use crate::test_support::csv_table;
use crate::{
    BinomialLink, Family, FitOptions, GammaLink, GroupIds, ModelSpec, PoissonLink, Re, SeFor,
};

const REL: f64 = 1e-12;

fn numeric<'a>(t: &'a Table, name: &str) -> &'a [f64] {
    match t.columns.iter().find(|(c, _)| c == name) {
        Some((_, Column::Numeric(v))) => v,
        _ => panic!("no numeric column {name}"),
    }
}

fn mean(v: impl Iterator<Item = f64>, n: usize) -> f64 {
    v.sum::<f64>() / n as f64
}

fn fit_formula(formula: &str, t: &Table, family: Family) -> (Lowered, Fit) {
    let lowered = formula::lower(formula, t, family).unwrap();
    let fit = crate::fit(&lowered.data(), lowered.spec(), &FitOptions::default()).unwrap();
    assert!(fit.converged(), "{formula}: not converged");
    (lowered, fit)
}

/// `var(Xβ̂)` over the rows, denominator `n − 1`.
fn fixed_variance(lowered: &Lowered, fit: &Fit) -> f64 {
    let (n, p) = (lowered.n, lowered.p);
    let eta: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| lowered.x[i * p + j] * fit.beta[j]).sum())
        .collect();
    let m = mean(eta.iter().copied(), n);
    eta.iter().map(|e| (e - m).powi(2)).sum::<f64>() / (n - 1) as f64
}

/// `fit`'s variance parts, R² and ICC on `lowered`'s data against the fixed,
/// random and residual variances `want`, each within `REL` of their total.
fn check(lowered: &Lowered, fit: &Fit, want: [f64; 3], what: &str) {
    let (data, spec) = (lowered.data(), lowered.spec());
    let got = fit.variance_parts(&data, spec).unwrap();
    let [f, r, e] = want;
    let total = f + r + e;
    let close = |g: f64, w: f64, scale: f64, name: &str| {
        assert!(
            (g - w).abs() <= REL * scale,
            "{what} {name}: got {g:e}, want {w:e}"
        );
    };
    close(got.fixed, f, total, "fixed");
    close(got.random, r, total, "random");
    close(got.residual, e, total, "residual");
    let r2 = fit.r2(&data, spec).unwrap();
    let icc = fit.icc(&data, spec).unwrap();
    close(r2.marginal, f / total, 1.0, "R2 marginal");
    close(r2.conditional, (f + r) / total, 1.0, "R2 conditional");
    close(icc.adjusted, r / (r + e), 1.0, "ICC adjusted");
    close(icc.unadjusted, r / total, 1.0, "ICC unadjusted");
}

fn sleepstudy() -> Table {
    csv_table(
        include_str!("../../validation/data/empirical/sleepstudy.csv"),
        &["Subject"],
    )
}

/// The random variance of one block on a slope variable `v`: the row mean of
/// `zᵢᵀ D zᵢ`, `zᵢ = (1, vᵢ)` with an intercept and `(vᵢ)` without.
fn block_variance(block: &[f64], v: &[f64]) -> f64 {
    let n = v.len();
    match block {
        [d] => d * mean(v.iter().map(|x| x * x), n),
        [d00, d10, d11] => mean(v.iter().map(|x| d00 + 2.0 * d10 * x + d11 * x * x), n),
        _ => panic!("block of {} entries", block.len()),
    }
}

#[test]
fn masked_and_re_only_blocks() {
    let t = sleepstudy();
    let days = numeric(&t, "Days");

    // `(Days || Subject)`: two terms on one grouping, covariance exactly 0.
    let (lo, fit) = fit_formula("Reaction ~ Days + (Days || Subject)", &t, Family::Gaussian);
    assert_eq!(fit.varcorr[0].len(), 3);
    assert_eq!(fit.varcorr[0][1], 0.0);
    let want = [
        fixed_variance(&lo, &fit),
        block_variance(&fit.varcorr[0], days),
        fit.dispersion,
    ];
    check(&lo, &fit, want, "(Days || Subject)");

    // `(0 + Days | Subject)`: the block reports the slope alone.
    let (lo, fit) = fit_formula(
        "Reaction ~ Days + (0 + Days | Subject)",
        &t,
        Family::Gaussian,
    );
    assert_eq!(fit.varcorr[0].len(), 1);
    let want = [
        fixed_variance(&lo, &fit),
        block_variance(&fit.varcorr[0], days),
        fit.dispersion,
    ];
    check(&lo, &fit, want, "(0 + Days | Subject)");

    // `y ~ 1 + (Days | Subject)`: the slope reads a random-effect-only column.
    let (lo, fit) = fit_formula("Reaction ~ 1 + (Days | Subject)", &t, Family::Gaussian);
    assert_eq!((lo.p, lo.re_p), (1, 1));
    let want = [
        fixed_variance(&lo, &fit),
        block_variance(&fit.varcorr[0], days),
        fit.dispersion,
    ];
    check(&lo, &fit, want, "1 + (Days | Subject)");
}

#[test]
fn binomial_probit_and_cloglog() {
    let mut t = csv_table(
        include_str!("../../validation/data/empirical/cbpp.csv"),
        &["herd", "period"],
    );
    let size = numeric(&t, "size").to_vec();
    let healthy: Vec<f64> = size
        .iter()
        .zip(numeric(&t, "incidence"))
        .map(|(s, i)| s - i)
        .collect();
    t.columns.push(("healthy".into(), Column::Numeric(healthy)));
    let w_bar = mean(size.iter().copied(), size.len());
    let formula = "cbind(incidence, healthy) ~ period + (1 | herd)";
    for (link, residual) in [
        (BinomialLink::Probit, 1.0 / w_bar),
        (BinomialLink::Cloglog, PI * PI / (6.0 * w_bar)),
    ] {
        let (lo, fit) = fit_formula(formula, &t, Family::Binomial { link });
        assert_eq!(lo.weights.as_deref(), Some(size.as_slice()));
        let want = [fixed_variance(&lo, &fit), fit.varcorr[0][0], residual];
        check(&lo, &fit, want, &format!("{link:?}"));
    }
}

#[test]
fn poisson_null_fit_gives_mu() {
    let t = csv_table(
        include_str!("../../validation/data/simulated/sim_poisson_offset.csv"),
        &["cluster"],
    );
    let family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let (lo, fit) = fit_formula("y ~ x + offset(log_exposure) + (1 | cluster)", &t, family);
    let offset = lo.offset.as_deref().expect("the formula's offset");

    // The null model by hand: the intercept, the same random intercept, ids
    // and offset.
    let ones = vec![1.0; lo.n];
    let null_data = Data::new(&ones, &lo.y, lo.n, 1).ids(&lo.ids).offset(offset);
    let opts = FitOptions {
        se_for: SeFor::Only(vec![]),
        ..FitOptions::default()
    };
    let null = crate::fit(
        &null_data,
        &ModelSpec::mixed(family, Re::intercept()),
        &opts,
    )
    .unwrap();
    assert!(null.converged());
    let mu = (null.beta[0] + 0.5 * null.varcorr[0][0]).exp();
    let residual = (1.0 / mu).ln_1p();

    let got = fit.variance_parts(&lo.data(), lo.spec()).unwrap();
    assert_eq!(got.residual.to_bits(), residual.to_bits());
    let want = [fixed_variance(&lo, &fit), fit.varcorr[0][0], residual];
    check(&lo, &fit, want, "Poisson with an offset");
}

/// A grouping with one level per row is insight's observation-level term: its
/// variance joins the residual and stays out of the null model's `μ`.
#[test]
fn observation_level_effect_joins_the_residual() {
    let mut t = csv_table(
        include_str!("../../validation/data/simulated/sim_poisson_offset.csv"),
        &["cluster"],
    );
    let obs: Vec<String> = (0..t.n).map(|i| format!("o{i}")).collect();
    t.columns
        .push(("obs".into(), Column::factor_from_labels(&obs)));
    let family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let (lo, fit) = fit_formula(
        "y ~ x + offset(log_exposure) + (1 | cluster) + (1 | obs)",
        &t,
        family,
    );
    let obs_var = fit.varcorr[1][0];
    assert!(
        obs_var > 0.0,
        "the observation-level variance is {obs_var:e}"
    );

    let ones = vec![1.0; lo.n];
    let offset = lo.offset.as_deref().expect("the formula's offset");
    let null_data = Data::new(&ones, &lo.y, lo.n, 1).ids(&lo.ids).offset(offset);
    let opts = FitOptions {
        se_for: SeFor::Only(vec![]),
        ..FitOptions::default()
    };
    let null = crate::fit(&null_data, lo.spec(), &opts).unwrap();
    assert!(null.converged());
    let mu = (null.beta[0] + 0.5 * null.varcorr[0][0]).exp();
    let want = [
        fixed_variance(&lo, &fit),
        fit.varcorr[0][0],
        (1.0 / mu).ln_1p() + obs_var,
    ];
    check(&lo, &fit, want, "Poisson with an observation-level term");
}

#[test]
fn zero_random_variance() {
    let t = csv_table(
        include_str!("../../validation/data/empirical/Dyestuff2.csv"),
        &["Batch"],
    );
    let (lo, fit) = fit_formula("Yield ~ 1 + (1 | Batch)", &t, Family::Gaussian);
    assert_eq!(fit.varcorr[0], vec![0.0], "Dyestuff2 is singular");
    let (data, spec) = (lo.data(), lo.spec());
    let r2 = fit.r2(&data, spec).unwrap();
    let icc = fit.icc(&data, spec).unwrap();
    assert_eq!(icc.adjusted, 0.0);
    assert_eq!(icc.unadjusted, 0.0);
    assert_eq!(r2.conditional, r2.marginal);
}

#[test]
fn refusals() {
    let t = sleepstudy();
    let (fixed, fixed_fit) = fit_formula("Reaction ~ Days", &t, Family::Gaussian);

    let gamma = csv_table(
        include_str!("../../validation/data/simulated/sim_gamma.csv"),
        &["cluster", "grp"],
    );
    let inverse = formula::lower(
        "y ~ x + (1 | cluster)",
        &gamma,
        Family::Gamma {
            link: GammaLink::Inverse,
        },
    )
    .unwrap();
    let inverse_fit = crate::fit(&inverse.data(), inverse.spec(), &FitOptions::default()).unwrap();

    let (mixed, mixed_fit) = fit_formula("Reaction ~ Days + (1 | Subject)", &t, Family::Gaussian);
    let (n, p) = (mixed.n - 1, mixed.p);
    let short_ids = GroupIds::new(mixed.ids.primary[..n].to_vec());
    let short = Data::new(&mixed.x[..n * p], &mixed.y[..n], n, p).ids(&short_ids);

    let cases: [(&str, &Fit, Data<'_>, &ModelSpec, bool); 3] = [
        ("fixed-only", &fixed_fit, fixed.data(), fixed.spec(), true),
        (
            "Gamma inverse",
            &inverse_fit,
            inverse.data(),
            inverse.spec(),
            true,
        ),
        ("another n", &mixed_fit, short, mixed.spec(), false),
    ];
    for (what, fit, data, spec, unsupported) in cases {
        let refused = |e: FitError| match e {
            FitError::Unsupported(_) => assert!(unsupported, "{what}: {e:?}"),
            FitError::Shape(_) => assert!(!unsupported, "{what}: {e:?}"),
            e => panic!("{what}: {e:?}"),
        };
        refused(fit.r2(&data, spec).expect_err(what));
        refused(fit.icc(&data, spec).expect_err(what));
    }
}
