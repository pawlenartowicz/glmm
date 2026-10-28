//! Dispatch-level tests shared across estimators: entry checks and
//! marshalling (`FitError` variants, ignored `nagq`, rank-deficiency salvage),
//! θ-width/varcorr bookkeeping, `spec_sized_from_ids` sizing, and
//! `classify_design` routing. Also hosts the golden-data generators
//! (`lcg`, `lmm_hand_dataset`, `dense_ids`, `dense_str`, `sim_clustered`)
//! shared by 2+ estimator test files.

use super::common::check_model_shape;
use super::*;
use crate::standardize::ColumnScaling;
use crate::test_support::{fit_cold, fit_warm_wo, ids_from_sizing, Sizing};
use crate::{
    BinomialLink, Family, GammaLink, GroupIds, Grouping, GroupingRelation, InverseGaussianLink,
    ModelSpec, NegBinomialLink, PoissonLink, ReStructure, StartValues,
};

/// Band for pins on the closed-form paths — OLS and GLM/IRLS.
///
/// These are Rust-vs-Rust pins, NOT cross-engine agreement bands: they say the
/// number has not moved, and nothing about lme4. Cross-engine bands live in
/// `validation/grid/tol.R` and are asserted by the `oracle-tests` tier.
pub(crate) const PIN_REL_OLS: f64 = 1e-9;

/// Band for pins on the iterative paths — LMM/GLMM, where BOBYQA stops inside
/// its own convergence tolerance rather than on a fixed point, so a shifted
/// last bit in the objective moves the reported optimum further than it does on
/// the closed-form paths.
///
/// Measured, not chosen — see [`assert_pinned`] for which machine measured it
/// and what the second machine cost. Bit-exact pins would be the tighter claim
/// and are wrong for exactly that reason: they would go red on a host drawing a
/// different microarchitecture, for no bug.
pub(crate) const PIN_REL_ITER: f64 = 1e-7;

/// Assert a fitted vector against its pinned values, elementwise relative.
/// `what` names the quantity so a failure says which one moved. Length is
/// asserted too: a pin that silently covers fewer elements than the fit
/// produces is the same nothing-asserted failure the pins exist to prevent.
///
/// The failure names the WORST element, not the first one over the band. That
/// matters for the band-measurement run below: stopping at the first element
/// would report whichever came first in index order rather than the spread the
/// band has to cover.
///
/// An element passes only if it is within the band, so a NaN in `got` against
/// a finite pin fails. A NaN pin matches only a NaN (a quantity that is
/// undefined by construction, such as the SE of a zero stddev), and a pin with
/// no finite element at all fails: it would assert nothing. Equal values pass
/// outright, which covers exact-zero pins.
///
/// # Which machine the pins are frozen on
///
/// This is the one place that says it; every pin site cites this function rather
/// than re-explaining. If you are re-freezing a pin, read this first.
///
/// **The anchor is x86_64-unknown-linux-gnu**, Intel Core Ultra 7 265H (Arrow
/// Lake-H, AVX2 + FMA, no AVX-512). Every Rust-vs-Rust pin in the crate — the
/// `PIN_REL_OLS` / `PIN_REL_ITER` pins and the per-test `BAND` pins in
/// `glmm_tests.rs` and `sparse/tests.rs` — reproduces there BIT-EXACTLY, worst
/// relative spread 0.0, across all four feature configs (default,
/// `loop_advanced`, `parallel`, `--no-default-features`) in both debug and
/// release: eight legs, zero non-exact pins. Verified 2026-07-31 by rewriting
/// every band to `1e-300` and re-running the suite, so a bit-exact pin passes
/// and one off by a single ULP fails. (`1e-300` and not `0.0` because the
/// comparison was strict then; it is `rel <= band` now, so `0.0` does the
/// same.) That is the `pin-bands`
/// workflow's procedure, and it rewrites this band list — keep the two in step.
///
/// **The bands are margin for a different CPU, and that cost is now measured.**
/// The second machine is aarch64-apple-darwin, where `simd_transcendental.rs`
/// dispatches differently and the compiler contracts multiply-adds into FMA at
/// other points in this kernel's long reductions. Drift off the anchor values,
/// worst per test: 1.4e-7 (`fit_wide_crossed_sparse_is_pinned`), 1.5e-7
/// (`fit_glmm_binomial_slope1_vector_agq_is_pinned`), 5.1e-5
/// (`fit_wide_slopes_sparse_is_pinned`, its smallest `ge` covariance entry;
/// 1.4e-6 before the 0.5.0 re-pins, everything else in it ≤ 5.0e-6), 3.6e-6
/// (`fit_glmm_binomial_slope2_vector_agq_is_pinned`), 7.7e-6
/// (`fit_sparse_binomial_slope_crossed_is_pinned`), 7.1e-5
/// (`fit_glmm_poisson_slope1_vector_agq_is_pinned`), 7.9e-4
/// (`fit_sparse_nb_glmm_is_pinned`).
///
/// That four-order spread is not noise in the measurement — it tracks how much
/// ITERATION sits downstream of the reduction that rounds differently. A single
/// solve lands at ~1e-7; the sparse NB GLMM fit drifts by 7.9e-4. So a band is
/// sized per test against its own measured drift, never corpus-wide.
///
/// **Re-freezing rule: re-pin on the anchor.** A value harvested on the second
/// machine is as arithmetically valid as the anchor's, but mixing the two splits
/// the corpus across two references and makes "bit-exact on the anchor" stop
/// being checkable — which is the property that lets a real regression be told
/// apart from a port. Four `se` vectors were re-anchored on 2026-07-31 for
/// exactly that reason; each says so at its own site.
///
/// **Not covered by any of this: pins asserted against a frozen R/Julia
/// reference** (`fit_glmm_binomial_bigsd_agq_matches_lme4`, and everything in
/// `tests/validation_oracle.rs`). Those bands are cross-ENGINE agreement, sized
/// by the reference's own reproducibility, and re-pinning them to any machine of
/// ours would be re-pinning the oracle. They are never bit-exact and are not
/// supposed to be.
pub(crate) fn assert_pinned(got: &[f64], want: &[f64], band: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    assert!(
        want.iter().any(|w| !w.is_nan()),
        "{what}: every pinned value is NaN, nothing is asserted"
    );
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let rel = if g == w || (g.is_nan() && w.is_nan()) {
            0.0
        } else {
            (g - w).abs() / w.abs()
        };
        // NaN (one side NaN, or an infinite pin) ranks as the worst possible.
        let rel = if rel.is_nan() { f64::INFINITY } else { rel };
        if rel > worst.0 {
            worst = (rel, i);
        }
    }
    let (rel, i) = worst;
    assert!(
        rel <= band,
        "{what}[{i}] = {} vs pinned {} (rel {rel:.2e}, worst of {})",
        got[i],
        want[i],
        got.len()
    );
}

/// Uniform(0,1) via a 64-bit LCG (splitmix-style constants). Reproduces the
/// `+INF`-plateau cell bit for bit. `pub(crate)` so `src/sparse/tests.rs`
/// shares this generator with `src/fit/glmm_tests.rs` instead of each keeping
/// its own copy.
pub(crate) fn inf_plateau_lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
}

pub(crate) fn inf_plateau_normal(state: &mut u64) -> f64 {
    let (u1, u2) = (inf_plateau_lcg_next(state), inf_plateau_lcg_next(state));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

pub(crate) fn inf_plateau_exp1(state: &mut u64) -> f64 {
    -inf_plateau_lcg_next(state).ln()
}

/// Poisson(lambda) by the naive product-of-uniforms method — matches the
/// reproduction exactly, including its large-lambda normal-approximation
/// branch (never hit at this fixture's means).
pub(crate) fn inf_plateau_poisson(state: &mut u64, lambda: f64) -> f64 {
    if lambda > 200.0 {
        return (lambda + lambda.sqrt() * inf_plateau_normal(state))
            .max(0.0)
            .round();
    }
    let l = (-lambda).exp();
    let (mut k, mut prod) = (0.0f64, inf_plateau_lcg_next(state));
    while prod > l {
        k += 1.0;
        prod *= inf_plateau_lcg_next(state);
        if k > 1e6 {
            break;
        }
    }
    k
}

/// Parse a `cluster,x,grp,y` sim CSV → (X=[1,x,grp_b], y, dense cluster ids, n_clusters).
/// `pub(crate)` so `src/sparse/tests.rs` can load the same fixtures.
pub(crate) fn sim_clustered(csv: &str) -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        raw.push(f[0].parse().unwrap());
        x.extend_from_slice(&[
            1.0,
            f[1].parse().unwrap(),
            f64::from(u32::from(f[2] == "b")),
        ]);
        y.push(f[3].parse().unwrap());
    }
    let (ids, nc) = dense_ids(&raw);
    (x, y, ids, nc)
}

/// The error the stable entry returns on `data`; panics when the fit is
/// accepted (a `Fit` has no `Debug`, so `unwrap_err` is not available).
fn refusal(data: &Data<'_>, model: &ModelSpec, opts: &FitOptions) -> FitError {
    match crate::fit(data, model, opts) {
        Err(e) => e,
        Ok(_) => panic!("the input must be refused"),
    }
}

/// A wrong-length weights vector is refused on an otherwise-open path
/// (fixed-only OLS).
#[test]
fn weights_shape_still_asserted() {
    let n = 4;
    let x = vec![1.0f64; n];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let w = vec![1.0; n - 1];
    let e = refusal(
        &Data::new(&x, &y, n, 1).weights(&w),
        &model,
        &FitOptions::default(),
    );
    assert!(
        matches!(&e, FitError::Weights(m) if m.contains("weights must have n elements")),
        "{e:?}"
    );
}

/// Valid 4-row, 1-column fixed-only Gaussian OLS input, shared by the
/// entry-check tests below — each breaks exactly one field of it.
fn shape_check_fixture() -> (Vec<f64>, Vec<f64>, ModelSpec) {
    let n = 4;
    let x = vec![1.0f64; n];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    (x, y, model)
}

/// `y` shorter than `n` is refused at the entry, the same shape check as the
/// weights length above.
#[test]
fn wrong_y_length_is_a_shape_error() {
    let (x, _y, model) = shape_check_fixture();
    let y = vec![1.0, 2.0, 3.0];
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
    assert!(
        matches!(&e, FitError::Shape(m) if m.contains("y must have n elements")),
        "{e:?}"
    );
}

/// `x` short of `n*p` elements is refused at the entry, the same shape check
/// as the weights length above.
#[test]
fn wrong_x_length_is_a_shape_error() {
    let (x, y, model) = shape_check_fixture();
    let x = x[..x.len() - 1].to_vec();
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
    assert!(
        matches!(&e, FitError::Shape(m) if m.contains("n*p")),
        "{e:?}"
    );
}

/// A zero weight is not finite-and-positive, so it is refused the same way
/// as a wrong-length weights vector, naming its row.
#[test]
fn nonpositive_weight_is_rejected() {
    let (x, y, model) = shape_check_fixture();
    let w = [1.0, 0.0, 1.0, 1.0];
    let e = refusal(
        &Data::new(&x, &y, 4, 1).weights(&w),
        &model,
        &FitOptions::default(),
    );
    assert!(
        matches!(&e, FitError::Weights(m) if m.contains("weights") && m.contains("(row ")),
        "{e:?}"
    );
}

/// The offset finiteness check is family-independent, so a NaN offset is
/// refused on a negative-binomial GLM, naming its row — there is no
/// NB-specific rejection anywhere else on this route.
#[test]
fn nonfinite_offset_on_a_negative_binomial_glm_is_rejected() {
    let (x, y, mut model) = shape_check_fixture();
    model.family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let o = [0.0, f64::NAN, 0.0, 0.0];
    let e = refusal(
        &Data::new(&x, &y, 4, 1).offset(&o),
        &model,
        &FitOptions::default(),
    );
    assert!(
        matches!(&e, FitError::Offset(m) if m.contains("offset") && m.contains("(row ")),
        "{e:?}"
    );
}

/// A NaN or an infinite value in `x` is refused at the entry, naming its row
/// and column.
#[test]
fn nonfinite_x_is_rejected() {
    for bad in [f64::NAN, f64::INFINITY] {
        let (mut x, y, model) = shape_check_fixture();
        x[2] = bad;
        let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
        assert!(
            matches!(&e, FitError::Shape(m) if m == "x must be finite (row 2, column 0)"),
            "x = {bad}: {e:?}"
        );
    }
}

/// A NaN in `y` is refused at the entry as a response no family supports.
#[test]
fn nonfinite_y_nan_is_rejected() {
    let (x, mut y, model) = shape_check_fixture();
    y[1] = f64::NAN;
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
    assert!(
        matches!(&e, FitError::ResponseOutOfSupport(m) if m == "y must be finite (row 1)"),
        "{e:?}"
    );
}

/// A negative Poisson response is out of the family's domain and is refused
/// at the entry, before the kernel can converge on a meaningless answer (an
/// unrefused negative count reaches the kernel as `converged = true` with
/// logLik = −∞).
#[test]
fn poisson_negative_response_is_rejected() {
    let (x, mut y, mut model) = shape_check_fixture();
    model.family = Family::Poisson {
        link: PoissonLink::Log,
    };
    y[1] = -1.0;
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
    assert!(
        matches!(&e, FitError::ResponseOutOfSupport(m)
            if m.contains("Poisson response must be >= 0 (row 1)")),
        "{e:?}"
    );
}

/// Same domain check as Poisson above, on the negative-binomial GLM route.
#[test]
fn nb_negative_response_is_rejected() {
    let (x, mut y, mut model) = shape_check_fixture();
    model.family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    y[1] = -1.0;
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
    assert!(
        matches!(&e, FitError::ResponseOutOfSupport(m)
            if m.contains("negative-binomial response must be >= 0")),
        "{e:?}"
    );
}

/// A binomial response above 1 or below 0 is out of the success-proportion
/// domain and is refused at the entry — an unrefused one fits with
/// `converged = true` and a NaN logLik.
#[test]
fn binomial_response_outside_zero_one_is_rejected() {
    let (x, _, mut model) = shape_check_fixture();
    model.family = Family::Binomial {
        link: BinomialLink::Logit,
    };
    for y in [vec![0.0, 1.0, 2.0, 0.0], vec![0.0, -0.1, 1.0, 0.0]] {
        let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
        assert!(
            matches!(&e, FitError::ResponseOutOfSupport(m)
                if m.contains("binomial response (the success proportion) must be in [0, 1]")),
            "y = {y:?}: {e:?}"
        );
    }
}

/// A non-positive Gamma response is out of domain and is refused at the
/// entry — an unrefused one gives "Fit failed … simpler random effects /
/// rescale predictors", advice that has nothing to do with the actual problem.
#[test]
fn gamma_nonpositive_response_is_rejected() {
    let (x, mut y, mut model) = shape_check_fixture();
    model.family = Family::Gamma {
        link: GammaLink::Log,
    };
    y[1] = 0.0;
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
    assert!(
        matches!(&e, FitError::ResponseOutOfSupport(m) if m.contains("Gamma response must be > 0")),
        "{e:?}"
    );
}

/// Same domain check as Gamma above, on the inverse-Gaussian GLM route.
#[test]
fn inverse_gaussian_nonpositive_response_is_rejected() {
    let (x, mut y, mut model) = shape_check_fixture();
    model.family = Family::InverseGaussian {
        link: InverseGaussianLink::Log,
    };
    y[1] = -3.0;
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
    assert!(
        matches!(&e, FitError::ResponseOutOfSupport(m)
            if m.contains("inverse-Gaussian response must be > 0")),
        "{e:?}"
    );
}

/// A held dispersion that is not a finite positive number is not a
/// variance-family scale and is refused at the entry — an unrefused negative
/// one gives `converged = TRUE` with NaN SEs and logLik. `+∞` is the value
/// only the finiteness half of the check refuses.
#[test]
fn bad_dispersion_is_rejected() {
    let (x, y, mut model) = shape_check_fixture();
    model.family = Family::Gamma {
        link: GammaLink::Log,
    };
    for d in [-1.0, f64::NAN, f64::INFINITY] {
        let opts = FitOptions {
            dispersion: Some(d),
            ..FitOptions::default()
        };
        let e = refusal(&Data::new(&x, &y, 4, 1), &model, &opts);
        assert!(
            matches!(&e, FitError::OptionValue(m)
                if m == "FitOptions.dispersion must be finite and > 0"),
            "dispersion = {d}: {e:?}"
        );
    }
}

/// A held dispersion on a family that has none (Poisson's φ ≡ 1) is ignored,
/// per the field's own doc — the fit comes out identical to the same call
/// with `dispersion: None`.
#[test]
fn dispersion_on_a_family_without_one_is_ignored() {
    let (x, y, mut model) = shape_check_fixture();
    model.family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let without = fit_cold(
        &x,
        &y,
        4,
        1,
        &model,
        &GroupIds::default(),
        &FitOptions::default(),
    );
    let opts = FitOptions {
        dispersion: Some(2.0),
        ..FitOptions::default()
    };
    let with = fit_cold(&x, &y, 4, 1, &model, &GroupIds::default(), &opts);
    assert!(without.converged() && with.converged());
    assert_eq!(without.beta, with.beta);
    // `deviance` is NaN on every GLM fit by design (see `Fit::deviance`'s
    // doc); `loglik` is this route's real fitted criterion.
    assert_eq!(without.loglik, with.loglik);
}

/// A non-integer Poisson count still fits (the deviance and score are
/// continuous in `y`), but is reported through `Note::NonIntegerResponse`,
/// matching R's `poisson()$initialize` warning.
#[test]
fn poisson_non_integer_response_is_noted() {
    let (x, _, mut model) = shape_check_fixture();
    model.family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let y = vec![1.0, 2.5, 3.0, 4.0];
    let f = fit_cold(
        &x,
        &y,
        4,
        1,
        &model,
        &GroupIds::default(),
        &FitOptions::default(),
    );
    assert!(f.converged(), "expected a converged Poisson GLM fit");
    let notes: Vec<_> = f
        .diagnostics
        .notes
        .iter()
        .filter(|n| matches!(n, Note::NonIntegerResponse { .. }))
        .collect();
    assert_eq!(
        notes.len(),
        1,
        "expected one note, got {:?}",
        f.diagnostics.notes
    );
    assert!(matches!(notes[0], Note::NonIntegerResponse { rows: 1 }));
}

/// A binomial aggregated fit whose derived successes `y·wᵢ` are not integers
/// (rounding trial weights against a proportion) is noted the same way as a
/// non-integer Poisson count above, matching R's `binomial()$initialize`
/// warning.
#[test]
fn binomial_non_integer_successes_is_noted() {
    let (x, _y, mut model) = shape_check_fixture();
    model.family = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let y = vec![0.5, 0.5, 0.5, 0.6];
    let opts = FitOptions::default();
    let f = fit_warm_wo(
        &x,
        &y,
        4,
        1,
        &model,
        &GroupIds::default(),
        None,
        Some(&[2.0, 2.0, 2.0, 3.0]),
        None,
        &opts,
    );
    assert!(f.converged(), "expected a converged binomial GLM fit");
    let notes: Vec<_> = f
        .diagnostics
        .notes
        .iter()
        .filter(|n| matches!(n, Note::NonIntegerResponse { .. }))
        .collect();
    assert_eq!(
        notes.len(),
        1,
        "expected one note, got {:?}",
        f.diagnostics.notes
    );
    assert!(matches!(notes[0], Note::NonIntegerResponse { rows: 1 }));
}

/// A `StartValues.beta` neither empty nor exactly `p`-wide is refused
/// through `fit_with_start`, the one entry that takes a start.
#[test]
fn wrong_width_start_beta_is_rejected() {
    let (x, y, model) = shape_check_fixture();
    let p = 1;
    let start = StartValues {
        beta: vec![0.0; p + 1],
        theta: vec![],
        dispersion: None,
    };
    let data = Data::new(&x, &y, 4, p);
    let Err(e) = crate::fit_with_start(&data, &model, &start, &FitOptions::default()) else {
        panic!("the start must be refused");
    };
    assert!(
        matches!(&e, FitError::Start(m) if m.contains("beta")),
        "{e:?}"
    );
}

/// A model with random effects and a `Data` without ids is refused: the
/// level counts come from the ids, so there is nothing to size the random
/// effects by.
#[test]
fn mixed_spec_without_ids_is_an_ids_error() {
    let (x, y, _) = shape_check_fixture();
    let model = ModelSpec::mixed(Family::Gaussian, crate::Re::intercept());
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &FitOptions::default());
    assert!(matches!(&e, FitError::Ids(m) if m.contains("ids")), "{e:?}");
}

/// Ids attached to a model without random effects are refused rather than
/// ignored, since the caller then meant a model the spec does not describe;
/// empty ids count as none, and that fit goes through.
#[test]
fn fixed_spec_with_ids_is_an_ids_error() {
    let (x, y, model) = shape_check_fixture();
    let ids = GroupIds::new(vec![0, 0, 1, 1]);
    let e = refusal(
        &Data::new(&x, &y, 4, 1).ids(&ids),
        &model,
        &FitOptions::default(),
    );
    assert!(
        matches!(&e, FitError::Ids(m) if m.contains("without random effects")),
        "{e:?}"
    );
    let none = GroupIds::default();
    let fit = crate::fit(
        &Data::new(&x, &y, 4, 1).ids(&none),
        &model,
        &FitOptions::default(),
    );
    assert!(fit.is_ok_and(|f| f.converged()));
}

/// A `SeFor::Only` column past `p` is refused at the entry, naming the
/// column, instead of indexing past the design deep in a kernel.
#[test]
fn se_for_index_past_p_is_an_option_value_error() {
    let (x, y, model) = shape_check_fixture();
    let opts = FitOptions {
        se_for: SeFor::Only(vec![0, 7]),
        ..FitOptions::default()
    };
    let e = refusal(&Data::new(&x, &y, 4, 1), &model, &opts);
    assert!(
        matches!(&e, FitError::OptionValue(m) if m.contains("column 7")),
        "{e:?}"
    );
}

#[test]
fn fit_rank_deficient_drops_and_matches_reduced() {
    let n = 30;
    let p = 3;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = 1.0 + x1; // col2 = col0 + col1 exactly
        y[i] = 0.5 + 0.4 * x1 + 0.3 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions {
            se_for: SeFor::Only(vec![0, 1, 2]),
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "reduced OLS must converge");
    assert_eq!(
        f.aliased(),
        vec![false, false, true],
        "later collinear column dropped"
    );
    assert!(f.beta[2].is_nan(), "aliased β = NaN");
    assert!(f.se[2].is_nan(), "aliased se = NaN");
    // An aliased column has no covariance either: its whole `vcov` row and
    // column are NaN, while the kept block stays consistent with `se`.
    for i in 0..p {
        assert!(f.vcov[i][2].is_nan(), "aliased column keeps a NaN vcov col");
        assert!(f.vcov[2][i].is_nan(), "aliased column keeps a NaN vcov row");
    }
    assert_vcov_agrees_with_se(&f, p, "aliased");

    // Direct fit on the 2-column reduced design.
    let mut xr = vec![0.0f64; n * 2];
    for i in 0..n {
        xr[i * 2] = x[i * p];
        xr[i * 2 + 1] = x[i * p + 1];
    }
    let fr = fit_cold(
        &xr,
        &y,
        n,
        2,
        &model,
        &GroupIds::default(),
        &FitOptions {
            se_for: SeFor::Only(vec![0, 1]),
            ..FitOptions::default()
        },
    );
    assert!(
        (f.beta[0] - fr.beta[0]).abs() < 1e-9,
        "β0 {} vs reduced {}",
        f.beta[0],
        fr.beta[0]
    );
    assert!(
        (f.beta[1] - fr.beta[1]).abs() < 1e-9,
        "β1 {} vs reduced {}",
        f.beta[1],
        fr.beta[1]
    );
}

/// An aliased fixed column that is ALSO an RE random slope must report a
/// non-converged `Fit`, not panic.
///
/// `y ~ 1 + x1 + x2 + (0 + x2 | g)` with `x2 = 1 + x1` exactly, so `x2` is
/// aliased at `ALIAS_EPS` and `fit_checked` routes into `fit_rank_deficient` — which
/// then has to remap the RE slope index through the kept-columns map and finds
/// the slope's column gone. Dropping the random slope alongside the fixed column
/// would be a different model, so the model is genuinely unfittable; the question
/// is only how that is reported. Panicking here would take the caller's
/// whole process down — an R/Python user would get an abort instead of an
/// inspectable fit, and a loop caller would lose the entire run over one
/// degenerate draw.
///
/// Deliberately written with `catch_unwind` rather than `#[should_panic]`: the
/// assertion under test is that NO panic happens, so a regression must surface as
/// a test FAILURE with the panic message attached, not as an abort that takes the
/// rest of the suite's output with it. `AssertUnwindSafe` is sound here because
/// nothing observed after the call is shared with the closure — the inputs are
/// read-only and the `Fit` is moved out.
#[test]
fn rank_deficient_random_slope_returns_nonconverged_instead_of_panicking() {
    let (x, y, n, p, ids) = aliased_slope_design();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![2], // the aliased column, as a random slope
            extra_groupings: vec![],
        }),
    };
    let opts = FitOptions {
        se_for: SeFor::Only(vec![0, 1, 2]),
        ..FitOptions::default()
    };

    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_cold(&x, &y, n, p, &model, &ids, &opts)
    }));
    let f = match caught {
        Ok(f) => f,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            panic!("fit_cold panicked on a rank-deficient random slope: {msg}");
        }
    };

    assert!(
        !f.converged(),
        "an unfittable model must not report converged"
    );
    assert!(
        f.beta.iter().all(|b| b.is_nan()) && f.se.iter().all(|s| s.is_nan()),
        "β/se must be NaN-filled: beta {:?} se {:?}",
        f.beta,
        f.se
    );
    assert!(
        f.vcov.iter().all(|row| row.iter().all(|v| v.is_nan())),
        "vcov must be NaN-filled"
    );
    // The mask is the one diagnostic the caller can act on, so it is reported
    // rather than blanked — the same field the successful salvage fills.
    assert_eq!(
        f.aliased(),
        vec![false, false, true],
        "the aliased column is still named"
    );
    // θ width for q_p = 2 (intercept + one slope) is vech = 3; NaN-filled, not
    // dropped, because the RE structure was never reduced.
    assert_eq!(f.tau2.len(), 3, "tau2 keeps the unreduced θ width");
    assert!(f.tau2.iter().all(|t| t.is_nan()), "tau2 NaN-filled");
    assert!(f.dispersion.is_nan() && f.deviance.is_nan() && f.loglik.is_nan());
    assert_eq!(f.df, 0, "no parameters were estimated");
    assert!(
        !f.singular(),
        "singular is a fitted-boundary flag, not a failure flag"
    );
    assert!(f.varcorr.is_empty() && f.fitted.is_empty() && f.ranef.is_empty());
    // `reml` is the criterion the options resolve to, as on a fitted model.
    assert!(f.reml, "REML by default");
    let ml = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            reml: Some(false),
            ..opts
        },
    );
    assert!(!ml.converged() && !ml.reml, "ML on request");
}

/// `y ~ 1 + x1 + x2` over 8 clusters of 12 rows, with `x2 = 1 + x1` exactly,
/// so column 2 is aliased at `ALIAS_EPS` and column 1 is not.
fn aliased_slope_design() -> (Vec<f64>, Vec<f64>, usize, usize, GroupIds) {
    let (n_clusters, per) = (8usize, 12usize);
    let n = n_clusters * per;
    let p = 3usize;
    let mut st = 29u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = 1.0 + x1; // col2 = col0 + col1 exactly ⇒ aliased at 1e-14
        y[i] = 0.5 + 0.4 * x1 + 0.3 * lcg(&mut st);
    }
    let ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    (x, y, n, p, ids)
}

/// An intercept-free block reports its slopes only in `tau2` (`Fit::tau2`),
/// on the unfittable return as on a fitted model: `(0 + x2 | g)` on the
/// aliased column gives the same all-NaN `tau2` length as `(0 + x1 | g)`
/// fitted on the column that is not aliased.
#[test]
fn unfittable_intercept_free_random_slope_reports_the_slope_width() {
    let (x, y, n, p, ids) = aliased_slope_design();
    let model = |col: u32| ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: vec![crate::ReTerm::no_intercept().slopes([col])],
            slopes: vec![col],
            extra_groupings: vec![],
        }),
    };
    let opts = FitOptions::default();
    let fitted = fit_cold(&x, &y, n, p, &model(1), &ids, &opts);
    assert!(fitted.converged(), "the twin on column 1 must fit");
    assert_eq!(fitted.tau2.len(), 1, "one reported slope variance");
    let f = fit_cold(&x, &y, n, p, &model(2), &ids, &opts);
    assert!(!f.converged(), "an aliased slope is unfittable");
    assert_eq!(f.tau2.len(), fitted.tau2.len(), "tau2 {:?}", f.tau2);
    assert!(f.tau2.iter().all(|t| t.is_nan()), "tau2 NaN-filled");
}

/// The degenerate-shape GLMM return (`n <= p`) reports an intercept-free
/// block's slopes only in `tau2`, as a fitted model does: one slope
/// variance, not the block's three θ entries.
#[test]
fn degenerate_glmm_intercept_free_block_reports_the_slope_width() {
    let (n, p) = (3usize, 3usize);
    let x: Vec<f64> = (0..n * p).map(|k| (k % 5) as f64 + 0.5).collect();
    let y = vec![1.0, 2.0, 0.0];
    let ids = GroupIds {
        primary: vec![0, 1, 1],
        extra: vec![],
    };
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            terms: vec![crate::ReTerm::no_intercept().slopes([1u32])],
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let f = fit_cold(&x, &y, n, p, &model, &ids, &FitOptions::default());
    assert!(!f.converged() && f.beta.iter().all(|b| b.is_nan()));
    assert_eq!(f.tau2.len(), 1, "tau2 {:?}", f.tau2);
    assert!(f.tau2.iter().all(|t| t.is_nan()), "tau2 NaN-filled");
}

/// Rank-deficiency salvage on a fixed-only design: near-collinear
/// `y ~ 1 + x1 + x2 + x3` (x3 ≈ x1+x2) must drop x3, mark it in `Fit::aliased`,
/// and fit the reduced model.
///
/// Values recorded from glmm. They are validated by `sim_collinear_glm`, whose
/// cross-engine cell checks the same fit against R's `stats::glm` — including
/// that R drops the SAME column (R keeps the name and writes `NA`; glmm keeps
/// the column and writes NaN). That agreement claim lives there, not here, so
/// this test can pin at a band the closed-form OLS path actually holds.
#[test]
fn fit_sim_collinear_drops_the_aliased_column() {
    // Surviving coefficients of the reduced fit; index 3 is the dropped x3.
    const REF_BETA: [f64; 3] = [0.9521742640860978, 0.7192032159906663, -0.43352501721286113];
    const REF_SE: [f64; 3] = [0.05307754868563462, 0.0526001540491167, 0.05132117947475763];

    let csv = include_str!("../../validation/data/simulated/sim_collinear.csv");
    let mut y = Vec::<f64>::new();
    let mut cols: Vec<[f64; 3]> = Vec::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        cols.push([
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        ]);
    }
    let n = y.len();
    let p = 4; // intercept + x1 + x2 + x3
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = cols[i][0];
        x[i * p + 2] = cols[i][1];
        x[i * p + 3] = cols[i][2];
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions {
            se_for: SeFor::Only(vec![0, 1, 2, 3]),
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "reduced fit converges");
    assert_eq!(
        f.aliased(),
        vec![false, false, false, true],
        "x3 is the dependent column and the only one dropped"
    );
    assert!(f.beta[3].is_nan(), "aliased β = NaN");
    assert!(f.se[3].is_nan(), "aliased se = NaN");
    assert_pinned(&f.beta[..3], &REF_BETA, PIN_REL_OLS, "beta");
    assert_pinned(&f.se[..3], &REF_SE, PIN_REL_OLS, "se");
}

/// A design with two columns entangled to `δ = 2⁻ᵏ`, `[1, a, a + δ·b]`, and
/// the same model on well-conditioned columns, `[1, a, b]` (row-major, `n`
/// rows, `p = 3`). `a` and `b` are integers below 64 in magnitude, so
/// `a + δ·b` is exact in f64 for `k ≤ 40` and the two designs span exactly the
/// same space: the coefficient of `a + δ·b` is the `[1, a, b]` fit's `b`
/// coefficient divided by `δ`, and so is its SE, with no rounding in the map.
/// At `k = 20`, `cond(X)` after standardization is about 2e6, so a Gram of `X`
/// loses about 1e-4 relative where the reparameterized fit is exact to
/// round-off; the pivot ratio stays near 1e-12, above the alias gate's
/// `ALIAS_EPS`, so nothing is dropped. Each step of `k` quarters the ratio.
pub(crate) fn entangled_dyadic_designs(n: usize, k: i32) -> (Vec<f64>, Vec<f64>, f64) {
    let delta = 2f64.powi(-k);
    let mut x = Vec::with_capacity(3 * n);
    let mut xr = Vec::with_capacity(3 * n);
    for i in 0..n {
        let a = ((i * 37) % 101) as f64 - 50.0;
        let b = ((i * 53) % 97) as f64 - 48.0;
        x.extend([1.0, a, a + delta * b]);
        xr.extend([1.0, a, b]);
    }
    (x, xr, delta)
}

/// Deterministic pseudo-data (NR LCG), uniform in (−1, 1). The shared LCG
/// fixture generator for the crate's unit tests, so fixtures seeded alike
/// draw the same stream. Mirrored in R by `tests/fixtures/gen_crossed_slope_lme4.R`
/// (change together).
pub(crate) fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
}

/// n=48, p=3, 6 clusters — same shape as `src/lmm/tests.rs`'s `hand_dataset`, adapted
/// to the row-major f64 layout the friendly API expects.
pub(super) fn lmm_hand_dataset() -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n = 48usize;
    let p = 3;
    let n_clusters = 6usize;
    let mut st = 42u64;
    let u_c: Vec<f64> = (0..n_clusters).map(|_| 0.6 * lcg(&mut st)).collect();
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let c = i % n_clusters;
        let x1 = lcg(&mut st);
        let x2 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = x2;
        y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + u_c[c] + 0.8 * lcg(&mut st);
    }
    (x, y, n, p)
}

/// n=60, p=3 weighted-collinear OLS design: predictor columns 1 and 2
/// (`a` and `a + delta`) are identical on the first `split` rows and differ by
/// exactly 1 on the rest, and those differing rows carry weight `WSMALL`. Full
/// rank on the raw `X` (the alias gate passes it through untouched) and
/// near-singular on the weighted Gram `X'WX`, which is the case nothing
/// upstream of the solve can see. `WSMALL = 1e-11` puts the weighted pivot at
/// ~2e-13 — inside the flagging band and still positive-definite enough for
/// faer's Cholesky to accept it; a smaller weight makes `X'WX` numerically
/// indefinite and the route refuses on `llt` instead, which is a different
/// path. Shared by the ill-conditioned-flag pin (`FitDiagnostics`), the
/// `Note::IllConditioned` pin through the stable entry, and the honest-SE
/// report — each reads a different downstream field off the same design, so
/// the three must stay byte-identical or one could pass while silently no
/// longer testing what it claims to.
pub(crate) fn weighted_collinear_ols_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
    let (n, p, split) = (60usize, 3usize, 40usize);
    const WSMALL: f64 = 1e-11;
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let a = ((i * 13) % 17) as f64 - 8.0;
        let delta = if i < split { 0.0 } else { 1.0 };
        x.extend_from_slice(&[1.0, a, a + delta]);
        y.push(0.5 + 1.3 * a + 0.477 * (a + delta) + ((i % 3) as f64 - 1.0));
        w.push(if i < split { 1.0 } else { WSMALL });
    }
    (x, y, w, n, p)
}

#[test]
fn theta_width_counts_vech_blocks() {
    // intercept-only primary + 1 slope → q_p=2 → 3; one intercept-only crossed → +1.
    let re = ReStructure {
        terms: Vec::new(),
        slopes: vec![1],
        extra_groupings: vec![Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::Crossed,
            slopes: vec![],
        }],
    };
    assert_eq!(super::common::theta_width(Some(&re)), 3 + 1);
    assert_eq!(super::common::theta_width(None), 0);
}

fn fit_with_varcorr(vech: Vec<f64>) -> Fit {
    Fit {
        beta: vec![],
        se: vec![],
        vcov: vec![],
        tau2: vec![],
        dispersion: 1.0,
        diagnostics: crate::Diagnostics::from_flags(true, false, 0),
        varcorr: vec![vech],
        stddev_se: vec![],
        n_eval: 0,
        #[cfg(feature = "counters")]
        counters: crate::counters::EvalCounters::new(),
        deviance: f64::NAN,
        loglik: f64::NAN,
        df: 0,
        reml: false,
        nobs: 0,
        objective: crate::inference::Objective::Laplace,
        fitted: vec![],
        ranef: vec![],
        ranef_levels: vec![],
        condvar: vec![],
        condvar_x: vec![],
        condvar_cross: None,
        coef_df: vec![],
        satterthwaite: None,
        design_summary: None,
    }
}

/// q=3 hand-computed, chosen specifically to catch a vech-indexing bug:
/// D = [[4,1,2],[1,9,3],[2,3,16]] (sd = [2,3,4], all off-diagonal terms
/// distinct so a transposed/misindexed vech would mismatch). Column-major
/// lower-tri vech walk: c=0 → (D00,D10,D20)=(4,1,2); c=1 → (D11,D21)=(9,3);
/// c=2 → (D22)=(16). vech = [4,1,2,9,3,16], len=6 ⇒ q=3.
///
/// Also covers q=1, the common random-intercept case and the only length-1
/// vech: a scalar block has no off-diagonal, stddev is sqrt(variance) and the
/// 1x1 correlation matrix is [[1.0]].
#[test]
fn stddev_corr_q3_hand_math() {
    let (sd1, corr1) = fit_with_varcorr(vec![9.0]).stddev_corr(0);
    assert_eq!(sd1, vec![3.0]);
    assert_eq!(corr1, vec![vec![1.0]]);

    let f = fit_with_varcorr(vec![4.0, 1.0, 2.0, 9.0, 3.0, 16.0]);
    let (sd, corr) = f.stddev_corr(0);
    assert_eq!(sd, vec![2.0, 3.0, 4.0]);
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        assert_eq!(corr[i][i], 1.0);
    }
    // corr(0,1) = D10/(sd0*sd1) = 1/(2*3)
    assert!((corr[0][1] - 1.0 / 6.0).abs() < 1e-12);
    assert!((corr[1][0] - 1.0 / 6.0).abs() < 1e-12);
    // corr(0,2) = D20/(sd0*sd2) = 2/(2*4) = 0.25
    assert!((corr[0][2] - 0.25).abs() < 1e-12);
    assert!((corr[2][0] - 0.25).abs() < 1e-12);
    // corr(1,2) = D21/(sd1*sd2) = 3/(3*4) = 0.25
    assert!((corr[1][2] - 0.25).abs() < 1e-12);
    assert!((corr[2][1] - 0.25).abs() < 1e-12);
}

/// assemble_varcorr emits one block per grouping in declaration order:
/// primary q=2 then one scalar extra (θ=0.7, q=1) → D=0.49. The primary's
/// col-major lower-tri vech(Λ) [2, 0.5, 1] is Λ=[[2,0],[0.5,1]], so
/// D=ΛΛ'=[[4,1],[1,1.25]] and vech(D) = [4, 1, 1.25]; every block is ×scale.
#[test]
fn assemble_varcorr_one_block_per_grouping() {
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(
        &SizedSpec {
            spec: ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: vec![1],
                    extra_groupings: vec![Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::Crossed,
                        slopes: vec![],
                    }],
                }),
            },
            n_primary: 4,
            extra_counts: vec![3],
        },
        &[1],
        &[],
    );
    let theta = [2.0, 0.5, 1.0, 0.7];
    let vc = super::common::assemble_varcorr(&theta, &g, 1.0, ColumnScaling::IDENTITY);
    assert_eq!(vc.len(), 2);
    assert_eq!(vc[0], vec![4.0, 1.0, 1.25]);
    assert!((vc[1][0] - 0.49).abs() < 1e-12, "extra D {}", vc[1][0]);
    let scaled = super::common::assemble_varcorr(&theta, &g, 3.0, ColumnScaling::IDENTITY);
    assert_eq!(scaled[0], vec![12.0, 3.0, 3.75]);
    assert!(
        (scaled[1][0] - 1.47).abs() < 1e-12,
        "scaled extra D {}",
        scaled[1][0]
    );
}

#[test]
fn spec_sized_from_ids_derives_counts() {
    let re = ReStructure {
        terms: Vec::new(),
        slopes: vec![],
        extra_groupings: vec![
            Grouping {
                terms: Vec::new(),
                relation: GroupingRelation::Crossed,
                slopes: vec![],
            },
            Grouping {
                terms: Vec::new(),
                relation: GroupingRelation::NestedWithin,
                slopes: vec![],
            },
        ],
    };
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(re),
    };
    let ids = GroupIds {
        primary: vec![0, 1, 2, 0, 1, 2], // 3 primary levels
        extra: vec![vec![0, 0, 1, 1, 2, 2], vec![0, 1, 2, 3, 4, 5]], // crossed 3, nested 6 children
    };
    let (sized, _ids, _perm) = super::spec_sized_from_ids(&model, &ids);
    assert_eq!(sized.n_primary, 3);
    // Crossed: 3 levels; nested: 6 children / 3 parents = 2 per parent.
    assert_eq!(sized.extra_counts, vec![3, 2]);
}

/// UNBALANCED nesting (3 parents with 3, 2 and 1 distinct children), the
/// WIDEST parent first in primary order (id 0 → 3 children `{0,1,2}`, id 1 → 2 `{3,4}`, id 2 → 1 `{5}`). The old
/// `children.div_ceil(n_primary)` formula computed `⌈6/3⌉ = 2` from the
/// global `max(extra)+1 = 6` — under-sizing parent 0's true 3-wide block
/// because the fullest parent isn't the one that sets the global max id.
/// The true per-parent count (grouping rows by `(primary, extra)` pairs)
/// gets this right regardless of which parent is fullest.
/// The mirror layout, widest parent last (`0 → {0}`, `1 → {3,4}`, `2 → {6,7,8}`), is checked too.
#[test]
fn spec_sized_from_ids_nested_unbalanced_first_parent_widest() {
    let re = ReStructure {
        terms: Vec::new(),
        slopes: vec![],
        extra_groupings: vec![Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::NestedWithin,
            slopes: vec![],
        }],
    };
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(re),
    };
    // Widest parent first (where the old `div_ceil` formula under-sized it)
    // and widest parent last (padded contiguous ids, as `grouping_ids` emits):
    // the width is the true per-parent maximum either way.
    for ids in [
        GroupIds {
            primary: vec![0, 0, 0, 1, 1, 2],
            extra: vec![vec![0, 1, 2, 3, 4, 5]],
        },
        GroupIds {
            primary: vec![0, 1, 1, 2, 2, 2],
            extra: vec![vec![0, 3, 4, 6, 7, 8]],
        },
    ] {
        let (sized, _ids, _perm) = super::spec_sized_from_ids(&model, &ids);
        assert_eq!(sized.n_primary, 3, "{:?}", ids.primary);
        assert_eq!(sized.extra_counts, vec![3], "{:?}", ids.primary);
    }
}

/// A `q_g = 5` (intercept + 4 slopes) extra grouping is over the `MAX_EXTRA_Q = 4`
/// NoZ-scratch envelope and routes to Sparse. The sparse numeric path
/// makes this a *supported* design — it routes to Sparse and
/// `fit_cold` runs the sparse solver (returning a degenerate non-converged `Fit`
/// on this n=0 input) instead of hitting the removed stub.
#[test]
fn fit_extra_grouping_q_too_large_routes_sparse() {
    let model = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q
                }],
            }),
        },
        n_primary: 4,
        extra_counts: vec![4],
    };
    assert!(matches!(classify_design(&model, 1), Solver::Sparse));
    // n=0: bypasses rank-deficiency detection (guarded by `if n > 0 && p > 0`)
    // so routing reaches classify_design → Sparse → the LMM's sparse kernel,
    // which runs (no PD data at n=0 ⇒ non-converged) rather than panicking.
    let n = 0;
    let p = 5; // slopes [1,2,3,4] in-bounds; q_g=5 over the NoZ envelope
    let fit = fit_cold(
        &[],
        &[],
        n,
        p,
        &model.spec,
        &ids_from_sizing(
            &model.spec,
            &Sizing::FixedClusters { n_clusters: 4 },
            &[4],
            n,
        ),
        &FitOptions {
            se_for: SeFor::Only(vec![1]),
            ..FitOptions::default()
        },
    );
    assert!(!fit.converged());
}

/// Anti-panic floor: an over-envelope mixed model must return a `Fit`, never
/// panic — for both over-envelope shapes (over-count: 7 crossed extras >
/// MAX_EXTRA_GROUPINGS; over-width: a q_g=5 slope block > MAX_EXTRA_Q) across
/// every family. `classify_design` routes both shapes to Sparse, and the sparse
/// path builds its own cap-free structures (`SparseLmmWorkspace::new` does not
/// call `add_rows_multi`, and `from_cluster_spec_ext` carries no
/// `n_extras <= MAX_EXTRA_GROUPINGS` guard). The constant-`y` input is
/// degenerate, so the fits need not converge; they only have to return.
#[test]
fn fit_over_envelope_never_panics() {
    let families = [
        Family::Gaussian,
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        Family::Gamma {
            link: crate::GammaLink::Log,
        },
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
    ];
    for family in families {
        // Over-count: 7 crossed intercept-only extras.
        let extra_groupings: Vec<Grouping> = (0..7)
            .map(|_| Grouping {
                terms: Vec::new(),
                relation: GroupingRelation::Crossed,
                slopes: vec![],
            })
            .collect();
        let model = SizedSpec {
            spec: ModelSpec {
                family,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: vec![],
                    extra_groupings,
                }),
            },
            n_primary: 2,
            extra_counts: vec![2; 7],
        };
        assert!(matches!(classify_design(&model, 1), Solver::Sparse));
        let (n, p) = (8, 1);
        let x = vec![1.0f64; n * p];
        let y = vec![1.0f64; n];
        let ids = GroupIds {
            primary: vec![0; n],
            extra: vec![vec![0; n]; 7],
        };
        let fit = fit_cold(
            &x,
            &y,
            n,
            p,
            &model.spec,
            &ids,
            &FitOptions {
                se_for: SeFor::Only(vec![0]),
                ..FitOptions::default()
            },
        );
        assert_eq!(fit.beta.len(), p, "{family:?} over-count returns a Fit");

        // Over-width: one crossed extra with a width-5 slope block (q_g = 5).
        let model_w = SizedSpec {
            spec: ModelSpec {
                family,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: vec![],
                    extra_groupings: vec![Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::Crossed,
                        slopes: vec![1, 2, 3, 4],
                    }],
                }),
            },
            n_primary: 4,
            extra_counts: vec![4],
        };
        assert!(matches!(classify_design(&model_w, 1), Solver::Sparse));
        let (n, p) = (16, 5);
        let mut st = 11u64;
        let x: Vec<f64> = (0..n)
            .flat_map(|_| {
                let mut r = [0.0f64; 5];
                r[0] = 1.0;
                for v in r[1..].iter_mut() {
                    *v = lcg(&mut st);
                }
                r
            })
            .collect();
        let y = vec![1.0f64; n];
        let ids = GroupIds {
            primary: (0..n as u32).map(|i| i % 4).collect(),
            extra: vec![(0..n as u32).map(|i| (i / 4) % 4).collect()],
        };
        let fit = fit_cold(
            &x,
            &y,
            n,
            p,
            &model_w.spec,
            &ids,
            &FitOptions {
                se_for: SeFor::Only(vec![1]),
                ..FitOptions::default()
            },
        );
        assert_eq!(fit.beta.len(), p, "{family:?} over-width returns a Fit");
    }
}

/// Degenerate shapes on the mixed non-Gaussian route: `p == 0` (no fixed
/// columns at all), `n == 0` and `n <= p` must return a non-converged `Fit`,
/// never panic — for the in-envelope (dense) layout, the over-envelope (7
/// crossed extras, routes `Sparse`/`Packed`) layout, and the `n == 0` shape,
/// across the wired non-Gaussian families. The all-zero-column case is the
/// width-0 re-entry: the column is aliased, so `fit_rank_deficient` re-enters
/// the fit with no column kept. The last case is `n == p`.
#[test]
fn fit_glmm_degenerate_width_never_panics() {
    let families = [
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        Family::Gamma {
            link: crate::GammaLink::Log,
        },
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
    ];
    for family in families {
        // p == 0, in-envelope (no extra groupings).
        let model = ModelSpec {
            family,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let (n, p) = (4, 0);
        let ids = GroupIds {
            primary: vec![0, 1, 0, 1],
            extra: vec![],
        };
        let fit = fit_cold(
            &[],
            &vec![1.0f64; n],
            n,
            p,
            &model,
            &ids,
            &FitOptions::default(),
        );
        assert!(
            !fit.converged(),
            "{family:?} p=0 in-envelope: degenerate fit does not report converged"
        );

        // p == 0, over-envelope (7 crossed extras > MAX_EXTRA_GROUPINGS).
        let extra_groupings: Vec<Grouping> = (0..7)
            .map(|_| Grouping {
                terms: Vec::new(),
                relation: GroupingRelation::Crossed,
                slopes: vec![],
            })
            .collect();
        let model_over = SizedSpec {
            spec: ModelSpec {
                family,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: vec![],
                    extra_groupings,
                }),
            },
            n_primary: 2,
            extra_counts: vec![2; 7],
        };
        assert!(matches!(classify_design(&model_over, 1), Solver::Sparse));
        let (n, p) = (8, 0);
        let ids = GroupIds {
            primary: vec![0; n],
            extra: vec![vec![0; n]; 7],
        };
        let fit = fit_cold(
            &[],
            &vec![1.0f64; n],
            n,
            p,
            &model_over.spec,
            &ids,
            &FitOptions::default(),
        );
        assert!(
            !fit.converged(),
            "{family:?} p=0 over-envelope: degenerate fit does not report converged"
        );

        // n == 0, p == 1.
        let ids_n0 = GroupIds {
            primary: vec![],
            extra: vec![],
        };
        let fit = fit_cold(&[], &[], 0, 1, &model, &ids_n0, &FitOptions::default());
        assert!(
            !fit.converged(),
            "{family:?} n=0: degenerate fit does not report converged"
        );

        // p == 1 with the one fixed column all zero.
        let n = 4;
        let ids = GroupIds {
            primary: vec![0, 1, 0, 1],
            extra: vec![],
        };
        let fit = fit_cold(
            &vec![0.0f64; n],
            &vec![1.0f64; n],
            n,
            1,
            &model,
            &ids,
            &FitOptions::default(),
        );
        assert!(
            !fit.converged(),
            "{family:?} all-zero column: degenerate fit does not report converged"
        );

        // n == p: four independent columns (lower-triangular ones) on four rows.
        let (n, p) = (4, 4);
        let x: Vec<f64> = (0..n * p)
            .map(|t| if t % p <= t / p { 1.0 } else { 0.0 })
            .collect();
        let fit = fit_cold(
            &x,
            &vec![1.0f64; n],
            n,
            p,
            &model,
            &ids,
            &FitOptions::default(),
        );
        assert!(
            !fit.converged() && fit.beta.iter().all(|b| b.is_nan()),
            "{family:?} n == p: degenerate fit is NaN and does not report converged"
        );
        assert_eq!(
            (fit.nobs, fit.objective),
            (n, Objective::Laplace),
            "{family:?} n == p"
        );
    }
}

/// Warm-path start-independence: on an LMM the MLE is start-independent, so a
/// warm fit from a perturbed `StartValues.theta` reaches the same β̂ as the cold
/// fit to optimizer tolerance — a good start shortens the path without moving the
/// MLE. n_theta=1 (intercept-only 6-cluster), so theta=[5.0] is
/// well off the THETA0 blind start.
///
/// Run once with a full β start and once from a θ-only start (`beta` empty —
/// the per-component cold marker the ports need for lme4's
/// `start = list(theta = …)`). The LMM ignores a β start anyway (β is solved
/// exactly given θ), so the empty row pins that an empty β is accepted rather
/// than faulting the entry assert, and that θ still threads through.
#[test]
fn fit_warm_start_reaches_cold_beta() {
    let (x, y, n, p) = lmm_hand_dataset();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = ids_from_sizing(&model, &Sizing::FixedClusters { n_clusters: 6 }, &[], n);
    let opts = FitOptions {
        se_for: SeFor::Only(vec![1, 2]),
        ..FitOptions::default()
    };
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    for beta in [vec![0.0; p], vec![]] {
        let start = StartValues {
            beta,
            theta: vec![5.0],
            dispersion: None,
        };
        let warm = fit_warm_wo(&x, &y, n, p, &model, &ids, Some(&start), None, None, &opts);
        assert!(
            cold.converged() && warm.converged(),
            "both fits must converge (start beta {:?})",
            start.beta
        );
        // An ignored θ start would replay the cold search, eval for eval.
        assert_ne!(
            warm.n_eval, cold.n_eval,
            "the θ start must reach the search (start beta {:?})",
            start.beta
        );
        for j in [1usize, 2] {
            let (a, b) = (cold.beta[j], warm.beta[j]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-7 || d <= 1e-6 * a.abs().max(b.abs()),
                "LMM MLE must be start-independent: β[{j}] cold {a} vs warm {b} (start beta {:?})",
                start.beta
            );
        }
    }
}

/// Map raw cluster labels to dense 0-based ids (first-seen order) + the count.
pub(super) fn dense_ids(raw: &[u32]) -> (Vec<u32>, usize) {
    use std::collections::HashMap;
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut next = 0u32;
    let ids: Vec<u32> = raw
        .iter()
        .map(|&r| {
            *map.entry(r).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            })
        })
        .collect();
    (ids, next as usize)
}

/// Map string factor labels to dense 0-based ids (first-seen order) + the count.
pub(super) fn dense_str(raw: &[String]) -> (Vec<u32>, usize) {
    use std::collections::HashMap;
    let mut map: HashMap<String, u32> = HashMap::new();
    let mut next = 0u32;
    let ids: Vec<u32> = raw
        .iter()
        .map(|r| {
            *map.entry(r.clone()).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            })
        })
        .collect();
    (ids, next as usize)
}

/// In-envelope designs route NoZ (byte-identical fast path); over-envelope
/// route Sparse. The boundary is the cap edge.
#[test]
fn classify_routes_at_the_cap_edge() {
    // A scalar-intercept LMM (q_p=1, no extras) — deep in-envelope.
    let in_env = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![],
            }),
        },
        n_primary: 10,
        extra_counts: vec![],
    };
    assert!(matches!(
        super::classify_design(&in_env, 1),
        super::Solver::NoZ
    ));

    // Exactly MAX_EXTRA_GROUPINGS intercept-only crossed groupings (24 crossed
    // levels, far under MAX_CROSSED_LEVELS): at the cap ⇒ NoZ.
    let at_cap_count = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: (0..crate::consts::MAX_EXTRA_GROUPINGS)
                    .map(|_| Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::Crossed,
                        slopes: vec![],
                    })
                    .collect(),
            }),
        },
        n_primary: 10,
        extra_counts: vec![4],
    };
    assert!(matches!(
        super::classify_design(&at_cap_count, 1),
        super::Solver::NoZ
    ));
    // q_p = MAX_PRIMARY_Q exactly: at the cap ⇒ NoZ.
    let at_cap_q = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: (1..crate::consts::MAX_PRIMARY_Q as u32).collect(), // q_p = MAX_PRIMARY_Q
                extra_groupings: vec![],
            }),
        },
        n_primary: 10,
        extra_counts: vec![],
    };
    assert!(matches!(
        super::classify_design(&at_cap_q, 1),
        super::Solver::NoZ
    ));

    // MAX_EXTRA_GROUPINGS+1 crossed groupings — over-envelope ⇒ Sparse.
    let over = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: (0..(crate::consts::MAX_EXTRA_GROUPINGS + 1))
                    .map(|_| Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::Crossed,
                        slopes: vec![],
                    })
                    .collect(),
            }),
        },
        n_primary: 10,
        extra_counts: vec![4],
    };
    assert!(matches!(
        super::classify_design(&over, 1),
        super::Solver::Sparse
    ));

    // Wide primary slope block past MAX_PRIMARY_Q ⇒ Sparse.
    let wide = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: (1..=crate::consts::MAX_PRIMARY_Q as u32).collect(), // q_p = 1 + MAX_PRIMARY_Q
                extra_groupings: vec![],
            }),
        },
        n_primary: 10,
        extra_counts: vec![],
    };
    assert!(matches!(
        super::classify_design(&wide, 1),
        super::Solver::Sparse
    ));
}

/// Total `Crossed` level count past MAX_CROSSED_LEVELS routes Sparse (the
/// dense tail is cubic in the SUM of crossed levels, so two half-cap
/// factors trip it together); at the cap exactly, intercept-only crossed
/// extras stay NoZ. Nested extras don't count toward the sum (they live in
/// the per-family elimination path, not the dense tail).
#[test]
fn classify_routes_many_crossed_levels_to_sparse() {
    let cap = crate::consts::MAX_CROSSED_LEVELS as u32;
    let spec = |extras: Vec<(Grouping, u32)>| SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: extras.iter().map(|(g, _)| g.clone()).collect(),
            }),
        },
        n_primary: 10,
        extra_counts: extras.iter().map(|&(_, k)| k as usize).collect(),
    };
    let crossed = |n_clusters: u32| {
        let g = Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::Crossed,
            slopes: vec![],
        };
        (g, n_clusters)
    };
    // One factor over the cap ⇒ Sparse.
    let over = spec(vec![crossed(cap + 1)]);
    assert!(matches!(
        super::classify_design(&over, 1),
        super::Solver::Sparse
    ));
    // Sum over factors trips the cap even when each is under it.
    let sum_over = spec(vec![crossed(cap / 2 + 1), crossed(cap / 2 + 1)]);
    assert!(matches!(
        super::classify_design(&sum_over, 1),
        super::Solver::Sparse
    ));
    // Exactly at the cap ⇒ NoZ unchanged.
    let at_cap = spec(vec![crossed(cap)]);
    assert!(matches!(
        super::classify_design(&at_cap, 1),
        super::Solver::NoZ
    ));
    // A many-level NESTED extra doesn't count toward the crossed sum.
    let nested = spec(vec![(
        Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::NestedWithin,
            slopes: vec![],
        },
        cap + 1,
    )]);
    assert!(matches!(
        super::classify_design(&nested, 1),
        super::Solver::NoZ
    ));
}

/// The measured q_g performance boundary for
/// Gaussian, and the only-implemented-route boundary for non-Gaussian: ANY
/// slope-carrying extra grouping routes Sparse. Gaussian intercept-only
/// extras (q_g = 1) stay NoZ (NoZ won 12–15× on the measured slice); the
/// dense NoZ GLMM kernel builds intercept-only extras exclusively, so for
/// non-Gaussian families Sparse is a correctness route, not a perf choice.
#[test]
fn classify_routes_slope_extras_to_sparse_all_families() {
    let spec = |family: Family, extra_slopes: Vec<u32>| SizedSpec {
        spec: ModelSpec {
            family,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: extra_slopes,
                }],
            }),
        },
        n_primary: 10,
        extra_counts: vec![5],
    };
    // Gaussian, slope-carrying extra (q_g = 2, in-envelope) ⇒ Sparse.
    let g_slope = spec(Family::Gaussian, vec![1]);
    assert!(matches!(
        super::classify_design(&g_slope, 1),
        super::Solver::Sparse
    ));
    // Gaussian, intercept-only extra ⇒ NoZ.
    let g_int = spec(Family::Gaussian, vec![]);
    assert!(matches!(
        super::classify_design(&g_int, 1),
        super::Solver::NoZ
    ));
    // Non-Gaussian, slope-carrying extra ⇒ Sparse (the only kernel that fits it).
    let p_slope = spec(
        Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        vec![1],
    );
    assert!(matches!(
        super::classify_design(&p_slope, 1),
        super::Solver::Sparse
    ));
}

/// A primary random slope column past `p` is refused at the entry, before any
/// workspace is built — the RE-envelope caps above route a design to
/// `Solver::Sparse`, but the column has to name a real design column on
/// every route regardless of which solver it reaches.
#[test]
fn out_of_range_primary_slope_column_is_rejected() {
    let n = 4;
    let p = 2;
    let x = vec![0.0f64; n * p];
    let y = vec![0.0f64; n];
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![7],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds::new(vec![0, 0, 1, 1]);
    let e = refusal(
        &Data::new(&x, &y, n, p).ids(&ids),
        &model,
        &FitOptions::default(),
    );
    assert!(
        matches!(&e, FitError::Shape(m) if m.contains("primary slope column 7")),
        "{e:?}"
    );
}

/// An extra-grouping random slope column past `p` is refused the same way as
/// the primary one above.
#[test]
fn out_of_range_extra_slope_column_is_rejected() {
    let n = 4;
    let p = 2;
    let x = vec![0.0f64; n * p];
    let y = vec![0.0f64; n];
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![],
            extra_groupings: vec![Grouping {
                terms: Vec::new(),
                relation: GroupingRelation::Crossed,
                slopes: vec![7],
            }],
        }),
    };
    let ids = GroupIds::new(vec![0, 0, 1, 1]).extra(vec![0, 1, 0, 1]);
    let e = refusal(
        &Data::new(&x, &y, n, p).ids(&ids),
        &model,
        &FitOptions::default(),
    );
    assert!(
        matches!(&e, FitError::Shape(m) if m.contains("extra-grouping slope column 7")),
        "{e:?}"
    );
}

/// Each refusal of the random-effect-only columns (`Data::re_columns`): a slope
/// index past `p + s`, a column no slope reads, the wrong length and a
/// non-finite entry, each a `FitError::Shape` naming the index or the cell.
#[test]
fn re_columns_shape_errors() {
    let (n, p) = (4, 2);
    let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y = vec![0.5, 1.0, 1.5, 2.0];
    let ids = GroupIds::new(vec![0, 0, 1, 1]);
    let model = |slopes: Vec<u32>| ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes,
            extra_groupings: vec![],
        }),
    };
    let z1 = vec![-1.0, 1.0, 1.0, -1.0];
    let z2 = vec![-1.0, 2.0, 1.0, 2.0, 1.0, 3.0, -1.0, 3.0];
    let mut z_nan = z1.clone();
    z_nan[2] = f64::NAN;
    let cases: [(&[f64], usize, Vec<u32>, &str); 4] = [
        (
            &z1,
            1,
            vec![3],
            "primary slope column 3 out of range (p=2, s=1)",
        ),
        (
            &z2,
            2,
            vec![2],
            "re_columns column 1 (slope index 3) is read by no random slope",
        ),
        (
            &z1[..3],
            1,
            vec![2],
            "re_columns must have n*s elements in row-major layout",
        ),
        (
            &z_nan,
            1,
            vec![2],
            "re_columns must be finite (row 2, column 0)",
        ),
    ];
    let mut refused = 0;
    for (z, s, slopes, want) in cases {
        let data = Data::new(&x, &y, n, p).ids(&ids).re_columns(z, s);
        let e = refusal(&data, &model(slopes), &FitOptions::default());
        assert!(matches!(&e, FitError::Shape(m) if m == want), "{e:?}");
        refused += 1;
    }
    assert_eq!(refused, 4);
}

/// A binomial-logit GLMM with a single intercept-only grouping (4
/// clusters) — the nagq tests below each override exactly the field that
/// trips their own check.
fn nagq_check_spec() -> ModelSpec {
    ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![],
            extra_groupings: vec![],
        }),
    }
}

#[test]
fn nagq_even_rejected() {
    // nagq=4 is even, in range, on an otherwise legal model: only the parity check fires.
    let e = check_model_shape(&nagq_check_spec(), 4).unwrap_err();
    assert!(
        matches!(&e, FitError::OptionValue(m) if m.contains("must be odd in 1..=")),
        "{e:?}"
    );
}

#[test]
fn nagq_above_max_rejected() {
    // nagq=27 is odd but past MAX_NAGQ (25).
    let e = check_model_shape(&nagq_check_spec(), 27).unwrap_err();
    assert!(
        matches!(&e, FitError::Envelope(m) if m.contains("nagq") && m.contains("25")),
        "{e:?}"
    );
}

/// Bit patterns of the fields a `nagq` that is ignored must leave unmoved:
/// β̂, the deviance and the notes.
fn nagq_fields(f: &Fit) -> (Vec<u64>, u64, String) {
    (
        f.beta.iter().map(|b| b.to_bits()).collect(),
        f.deviance.to_bits(),
        format!("{:?}", f.diagnostics.notes),
    )
}

/// `nagq > 1` on a model without random effects has no integral to
/// approximate: the fit runs at Laplace's answer, bit for bit, with no note.
#[test]
fn nagq_on_a_fixed_only_model_is_ignored() {
    let (x, _, mut model) = shape_check_fixture();
    model.family = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let y = vec![0.0, 1.0, 1.0, 0.0];
    let data = Data::new(&x, &y, 4, 1);
    let at = |nagq| {
        let opts = FitOptions {
            nagq,
            ..FitOptions::default()
        };
        crate::fit(&data, &model, &opts).unwrap_or_else(|e| panic!("nagq={nagq}: {e}"))
    };
    assert_eq!(nagq_fields(&at(3)), nagq_fields(&at(1)));
}

/// `nagq > 1` on a Gaussian mixed model is ignored the same way: its marginal
/// likelihood is exact, so the fit is the `nagq = 1` fit, bit for bit.
#[test]
fn nagq_on_a_gaussian_lmm_is_ignored() {
    let (x, y, n, p) = lmm_hand_dataset();
    let ids = GroupIds::new((0..n).map(|i| (i % 6) as u32).collect());
    let mut model = nagq_check_spec();
    model.family = Family::Gaussian;
    let data = Data::new(&x, &y, n, p).ids(&ids);
    let at = |nagq| {
        let opts = FitOptions {
            nagq,
            ..FitOptions::default()
        };
        crate::fit(&data, &model, &opts).unwrap_or_else(|e| panic!("nagq={nagq}: {e}"))
    };
    let (three, one) = (at(3), at(1));
    assert!(one.converged());
    assert_eq!(nagq_fields(&three), nagq_fields(&one));
}

/// The negative-binomial dispersion θ sits outside the integral (a
/// per-row constant in u), so the binomial/Poisson AGQ kernels serve NB.
#[test]
fn nagq_on_a_negative_binomial_glmm_is_accepted() {
    let mut model = nagq_check_spec();
    model.family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    assert_eq!(check_model_shape(&model, 3), Ok(()));
}

#[test]
fn nagq_over_the_q_cap_is_rejected() {
    let mut model = nagq_check_spec();
    // q_p = 4, past the temporary q_p<=3 cap.
    model.re.as_mut().unwrap().slopes = vec![0, 1, 2];
    let e = check_model_shape(&model, 3).unwrap_err();
    assert!(
        matches!(&e, FitError::Unsupported(m) if m.contains("exceeds the temporary")),
        "{e:?}"
    );
}

/// Tier-0 short-circuit: a fixed-only
/// fit (no random effects, no θ to search) must run the direct GLM/OLS path and
/// enter the BOBYQA search ZERO times. `fit_cold`'s `(family, None)` dispatch arms
/// route straight to `fit_ols_prebuilt`/`fit_glm`/`fit_glm_nb`, each of which hard-sets
/// `n_eval: 0`; this pins that invariant end-to-end (through the whole cold path,
/// not just `classify_design`) across every wired fixed-only family, so a future
/// change that accidentally sent a no-Z model through the optimizer would trip it.
/// NB's outer θ↔β alternation is a golden-section search, not BOBYQA — it also
/// leaves `n_eval` at the inner IRLS 0.
#[test]
fn fixed_only_fit_runs_zero_bobyqa_evals() {
    let n = 24;
    let p = 2;
    let mut st = 3u64;
    let mut x = vec![0.0f64; n * p];
    let mut xv = vec![0.0f64; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        xv[i] = x1;
    }
    // Per-family responses in each family's valid range so the direct fit is a
    // real convergent fit, not a degenerate one — the n_eval==0 invariant holds
    // regardless, but a healthy fit makes the guard exercise the live path.
    let families: [(Family, Vec<f64>); 5] = [
        (
            Family::Gaussian,
            (0..n).map(|i| 0.5 + 0.4 * xv[i]).collect(),
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            (0..n).map(|i| f64::from(u32::from(xv[i] > 0.0))).collect(),
        ),
        (
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            (0..n).map(|i| f64::from(1 + (i % 4) as u32)).collect(),
        ),
        (
            Family::Gamma {
                link: crate::GammaLink::Log,
            },
            (0..n).map(|i| 1.0 + 0.5 * (xv[i] + 1.0)).collect(),
        ),
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            (0..n).map(|i| f64::from(1 + (i % 4) as u32)).collect(),
        ),
    ];
    for (family, y) in families {
        let model = ModelSpec { family, re: None };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                se_for: SeFor::Only(vec![0, 1]),
                ..FitOptions::default()
            },
        );
        assert_eq!(
            f.n_eval, 0,
            "{family:?} fixed-only fit must enter BOBYQA zero times"
        );
    }
}

// ---------------------------------------------------------------------------
// `Fit::vcov` — gating tests for the `targets`-subset carve-out.
// ---------------------------------------------------------------------------

/// `vcov` must be a symmetric p×p whose diagonal IS `se²`, on every estimator
/// path — no path may report a finite `se[j]` next to a
/// NaN `vcov[j][j]`. Covers OLS / GLM / LMM / GLMM (both `WaldSe` arms), since
/// each sources `vcov` from a different matrix.
fn assert_vcov_agrees_with_se(fit: &Fit, p: usize, ctx: &str) {
    assert_eq!(fit.vcov.len(), p, "{ctx}: vcov is not p rows");
    for row in &fit.vcov {
        assert_eq!(row.len(), p, "{ctx}: vcov is not p×p");
    }
    for j in 0..p {
        if fit.se[j].is_finite() {
            assert!(
                fit.vcov[j][j].is_finite(),
                "{ctx}: finite se[{j}] alongside NaN vcov[{j}][{j}]"
            );
            // se = sqrt(diag) — same quantity, so this is tight, not a tolerance.
            let want = fit.se[j] * fit.se[j];
            assert!(
                (fit.vcov[j][j] - want).abs() <= 1e-9 * want.abs().max(1e-12),
                "{ctx}: vcov[{j}][{j}] = {} vs se[{j}]² = {want}",
                fit.vcov[j][j]
            );
        }
    }
    for i in 0..p {
        for j in 0..p {
            if fit.vcov[i][j].is_finite() || fit.vcov[j][i].is_finite() {
                assert_eq!(fit.vcov[i][j], fit.vcov[j][i], "{ctx}: vcov not symmetric");
            }
        }
    }
}

#[test]
fn vcov_diagonal_is_se_squared_on_every_path() {
    let (x, y, n, p) = lmm_hand_dataset();
    let all: Vec<u32> = (0..p as u32).collect();
    let opts = FitOptions {
        se_for: SeFor::Only(all.clone()),
        ..FitOptions::default()
    };

    // OLS — Gaussian, no RE.
    let ols = fit_cold(
        &x,
        &y,
        n,
        p,
        &ModelSpec {
            family: Family::Gaussian,
            re: None,
        },
        &GroupIds::default(),
        &opts,
    );
    assert!(ols.converged());
    assert_vcov_agrees_with_se(&ols, p, "ols");

    // LMM — Gaussian, 6 clusters.
    let ids = GroupIds {
        primary: (0..n).map(|i| (i % 6) as u32).collect(),
        extra: vec![],
    };
    let lmm_model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let lmm = fit_cold(&x, &y, n, p, &lmm_model, &ids, &opts);
    assert!(lmm.converged());
    assert_vcov_agrees_with_se(&lmm, p, "lmm");

    // GLM / GLMM — Bernoulli response off the same design.
    let yb: Vec<f64> = y.iter().map(|&v| f64::from(v > 0.5)).collect();
    let binom = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let glm = fit_cold(
        &x,
        &yb,
        n,
        p,
        &ModelSpec {
            family: binom,
            re: None,
        },
        &GroupIds::default(),
        &opts,
    );
    assert!(glm.converged());
    assert_vcov_agrees_with_se(&glm, p, "glm");

    // GLMM on both SE arms — each sources vcov from a different matrix
    // (FD-Hessian β block vs Schur inverse).
    for wald_se in [WaldSe::Hessian, WaldSe::Rx] {
        let glmm = fit_cold(
            &x,
            &yb,
            n,
            p,
            &ModelSpec {
                family: binom,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
            },
            &ids,
            &FitOptions {
                se_for: SeFor::Only(all.clone()),
                wald_se,
                ..FitOptions::default()
            },
        );
        assert!(glmm.converged(), "glmm {wald_se:?} did not converge");
        assert_vcov_agrees_with_se(&glmm, p, &format!("glmm {wald_se:?}"));
    }
}

/// The sanctioned `targets=` carve-out (gate 2's one exception): under a target
/// subset, `vcov` is finite exactly on the target block and NaN outside it —
/// never a finite `se` next to a NaN variance, and never a fabricated
/// covariance for a coefficient whose variance was never computed.
#[test]
fn vcov_is_nan_outside_the_target_block() {
    let (x, y, n, p) = lmm_hand_dataset();
    let fit = fit_cold(
        &x,
        &y,
        n,
        p,
        &ModelSpec {
            family: Family::Gaussian,
            re: None,
        },
        &GroupIds::default(),
        &FitOptions {
            se_for: SeFor::Only(vec![2]), // only the last predictor
            ..FitOptions::default()
        },
    );
    assert!(fit.converged());
    assert!(fit.se[2].is_finite() && fit.vcov[2][2].is_finite());
    assert!(fit.se[0].is_nan() && fit.se[1].is_nan());
    for j in [0usize, 1] {
        for i in 0..p {
            assert!(
                fit.vcov[i][j].is_nan() && fit.vcov[j][i].is_nan(),
                "vcov must be NaN outside the target block at ({i},{j})"
            );
        }
    }
    assert_vcov_agrees_with_se(&fit, p, "targets subset");
}

// ---------------------------------------------------------------------------
// The public `Diagnostics` surface: the three moved fields, the two
// reshaped ones, and the notes channel.
// ---------------------------------------------------------------------------

/// `Boundary` at both ends of the range the dense LMM route can report:
/// the deterministic τ̂=0 pin fixture (`fit_lmm_weighted_boundary_matches_wls`'s
/// design, same construction) lands `AtBoundary`, lmm_hand_dataset() with `i % 6` grouping lands `Interior`.
/// Also pins the one place `singular` and `boundary` are NOT interchangeable —
/// `singular` additionally carries the post-hoc negligible-component check.
#[test]
fn diagnostics_boundary_reports_both_ends() {
    let n = 48usize;
    let n_clusters = 6usize;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        // ±0.8 cancels exactly per cluster ⇒ the between-cluster variance MLE
        // is 0 and the optimizer pins it there.
        let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
        y[i] = 0.5 + 0.4 * x1 + e;
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let opts = FitOptions {
        se_for: SeFor::Only(vec![0, 1]),
        ..FitOptions::default()
    };
    let pinned = fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &GroupIds {
            primary: ids,
            extra: vec![],
        },
        &opts,
    );
    assert!(pinned.converged(), "a boundary fit still converges");
    assert_eq!(pinned.diagnostics.boundary, Boundary::AtBoundary);
    assert!(pinned.singular());
    assert_eq!(
        pinned.tau2,
        vec![0.0],
        "pin must be exact 0.0, not merely small"
    );
    assert!(pinned.beta[1].is_finite());

    let (xs, ys, ns, ps) = lmm_hand_dataset();
    let ids_s: Vec<u32> = (0..ns).map(|i| (i % 6) as u32).collect();
    let interior = fit_cold(
        &xs,
        &ys,
        ns,
        ps,
        &ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![],
            }),
        },
        &GroupIds {
            primary: ids_s,
            extra: vec![],
        },
        &FitOptions {
            se_for: SeFor::Only((0..ps as u32).collect()),
            ..FitOptions::default()
        },
    );
    assert!(interior.converged());
    assert_eq!(interior.diagnostics.boundary, Boundary::Interior);
    assert!(interior.diagnostics.pinned.is_empty(), "nothing pinned");
}

/// `pinned[g][i]` must pair with `stddev_corr(g).0[i]` — the whole point of the
/// alignment. Asserted on the one shape where a wrong
/// mapping is VISIBLE — a q=2 primary block where the SLOPE component pins and
/// the intercept stays interior, so swapping the two bits (or reading the vech
/// off-diagonal as a component) flips the assertion.
///
/// The design: x1 is a within-cluster antithetic ±1 pattern carrying a real
/// fixed slope but no cluster-varying one, and the ±0.8 period-4 residual is
/// in quadrature against it, so every cluster has Σresid = 0 and Σx1·resid = 0
/// exactly. The planted u₀ keeps the intercept component off the boundary, and
/// the quadrature keeps σ̂² positive: a residual collinear with x1 instead
/// sends σ̂² to 0 and the intercept θ to the box bound.
#[test]
fn diagnostics_pinned_aligns_with_varcorr_blocks() {
    let (nc, per) = (16usize, 16usize);
    let n = nc * per;
    let mut st = 5u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for (c, &uc) in u0.iter().enumerate() {
        for k in 0..per {
            let i = c * per + k;
            ids[i] = c as u32;
            let x1 = if k % 2 == 0 { 1.0 } else { -1.0 };
            let e = if (k / 2) % 2 == 0 { 0.8 } else { -0.8 };
            x[i * 2] = 1.0;
            x[i * 2 + 1] = x1;
            y[i] = 0.5 + 0.4 * x1 + uc + e;
        }
    }
    let fit = fit_cold(
        &x,
        &y,
        n,
        2,
        &ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        &GroupIds {
            primary: ids,
            extra: vec![],
        },
        &FitOptions {
            se_for: SeFor::Only(vec![0, 1]),
            ..FitOptions::default()
        },
    );
    assert!(fit.converged());
    assert_eq!(fit.diagnostics.pinned.len(), fit.varcorr.len());
    let (sd, _) = fit.stddev_corr(0);
    assert_eq!(fit.diagnostics.pinned[0].len(), sd.len());
    assert_eq!(
        fit.diagnostics.pinned[0],
        vec![false, true],
        "the SLOPE component is the pinned one; stddev {sd:?}"
    );
    // The pairing, stated as the invariant rather than as two literals: the
    // pinned slot's stddev collapses and the unpinned one's does not.
    //
    // NOT exactly zero, and that is not slack in the assertion. The pin fixes
    // the DIAGONAL θ (λ₁₁ = 0 exactly), while `stddev_corr(0).0[1]` is
    // √D₁₁ = √(λ₁₀² + λ₁₁²) — so it inherits whatever the off-diagonal λ₁₀
    // settled on, measured here at 4.4e-9 against a 0.6-scale intercept
    // component. A q≥2 pinned component reads as negligible, not as 0.0.
    assert!(
        sd[1] / sd[0] < 1e-6,
        "pinned component's stddev must collapse: {sd:?}"
    );
    assert!(sd[0] > 0.0, "interior component's stddev is positive");
    assert_eq!(fit.diagnostics.boundary, Boundary::AtBoundary);
    assert!(
        !fit.diagnostics
            .notes
            .iter()
            .any(|n| matches!(n, Note::VarianceAtLimit)),
        "the intercept component must be off the box bound: {:?}",
        fit.diagnostics.notes
    );
}

/// `Note::IllConditioned` through the STABLE entry, positive and negative.
/// Positive: `x = [1, a, a+δ]` where δ lives entirely on rows carrying weight
/// 1e-11 — full-rank raw (so the alias gate passes it through untouched) and
/// near-singular once weighted, which is the case nothing upstream of the fit
/// can see. Negative: the same design at unit weights raises nothing.
#[test]
fn diagnostics_ill_conditioned_note_through_fit_cold() {
    let (x, y, w, n, p) = weighted_collinear_ols_fixture();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let targets: Vec<u32> = (0..p as u32).collect();
    let flagged = fit_warm_wo(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        None,
        Some(&w),
        None,
        &FitOptions {
            se_for: SeFor::Only(targets.clone()),
            ..FitOptions::default()
        },
    );
    assert!(flagged.converged(), "fit-and-flag: the design is returned");
    assert_eq!(flagged.aliased(), vec![false; p], "nothing was dropped");
    assert_eq!(flagged.diagnostics.notes.len(), 1);
    let Note::IllConditioned { columns, pivot } = &flagged.diagnostics.notes[0] else {
        panic!(
            "expected IllConditioned, got {:?}",
            flagged.diagnostics.notes[0]
        );
    };
    assert_eq!(columns, &vec![2u32], "the later column of the pair");
    assert!(
        *pivot < crate::ols::PIVOT_MIN,
        "the note carries the measured ratio, got {pivot}"
    );

    let clean = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions {
            se_for: SeFor::Only(targets),
            ..FitOptions::default()
        },
    );
    assert!(clean.converged());
    assert!(
        clean.diagnostics.notes.is_empty(),
        "unweighted, the same design is well-conditioned"
    );
    assert!(clean.diagnostics.pinned.is_empty(), "no RE, nothing to pin");
}

/// The three GLMM-only carrier flags each become their own `Note`, through
/// `materialize_diagnostics` directly rather than a real GLMM fit: this pins the
/// mapping itself, one flag at a time, without depending on a fixture that
/// happens to trip both. The payloads a real solve produces are asserted
/// separately — `pirls_exhausted_note_counts_fit_path_evals_only` and
/// `exact_profile_fallback_reruns_on_pql_then_joint` in `glmm_tests`.
#[test]
fn carrier_flags_become_pirls_and_hessian_notes() {
    let d = super::common::FitDiagnostics {
        pirls_exhausted: 3,
        final_pirls_exhausted: true,
        hessian_fallback: true,
        exact_profile_fallback: true,
        ..super::common::FitDiagnostics::fixed_only(true)
    };
    let diag = super::common::materialize_diagnostics(&d, 0, &[]);
    assert_eq!(
        diag.notes.len(),
        3,
        "expected one PirlsExhausted, one HessianSeFallback and one ExactProfileFallback, got {:?}",
        diag.notes
    );
    let Note::PirlsExhausted { evals, final_eval } = &diag.notes[0] else {
        panic!("expected PirlsExhausted, got {:?}", diag.notes[0]);
    };
    assert_eq!(*evals, 3);
    assert!(*final_eval);
    assert!(
        matches!(diag.notes[1], Note::HessianSeFallback),
        "expected HessianSeFallback, got {:?}",
        diag.notes[1]
    );
    assert!(
        matches!(diag.notes[2], Note::ExactProfileFallback),
        "expected ExactProfileFallback, got {:?}",
        diag.notes[2]
    );
}

/// One frozen CSV fixture lowered through the formula frontend: `factors`
/// names the grouping columns, every other column is numeric.
#[cfg(feature = "formula")]
pub(crate) fn fixture_lowered(
    csv: &str,
    factors: &[&str],
    formula: &str,
    family: Family,
) -> crate::formula::Lowered {
    use crate::formula::{lower, Column, Table};
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim_matches('"').to_string())
        .collect();
    let rows: Vec<Vec<String>> = lines
        .map(|l| {
            l.split(',')
                .map(|s| s.trim_matches('"').to_string())
                .collect()
        })
        .collect();
    let columns = header
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let col = if factors.contains(&name.as_str()) {
                let labels: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
                Column::factor_from_labels(&labels)
            } else {
                Column::Numeric(rows.iter().map(|r| r[j].parse().unwrap()).collect())
            };
            (name.clone(), col)
        })
        .collect();
    lower(
        formula,
        &Table {
            n: rows.len(),
            columns,
        },
        family,
    )
    .unwrap()
}

/// `fit_rank_deficient` copies `nobs` and `objective` from its reduced fit (an
/// ML fit here, so the copy is not the REML default), and the unfittable
/// return carries the route's: the criterion on a Gaussian mixed model, the
/// quadrature on a GLMM.
#[test]
fn rank_deficient_fits_carry_nobs_and_the_route_objective() {
    let (n_clusters, per) = (8usize, 12usize);
    let n = n_clusters * per;
    let p = 3usize;
    let mut st = 29u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = 1.0 + x1; // col2 = col0 + col1 exactly
        y[i] = 0.5 + 0.4 * x1 + 0.3 * lcg(&mut st);
    }
    let yb: Vec<f64> = y.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
    let ids = GroupIds {
        primary: (0..n).map(|i| (i % n_clusters) as u32).collect(),
        extra: vec![],
    };
    let spec = |family: Family, slopes: Vec<u32>| ModelSpec {
        family,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes,
            extra_groupings: vec![],
        }),
    };
    let opts = |reml: Option<bool>, nagq: u8| FitOptions {
        se_for: SeFor::Only(vec![0, 1, 2]),
        reml,
        nagq,
        ..FitOptions::default()
    };

    // Salvaged: column 2 dropped, the reduced ML fit's values.
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &spec(Family::Gaussian, vec![]),
        &ids,
        &opts(Some(false), 1),
    );
    assert_eq!(f.aliased(), vec![false, false, true]);
    assert_eq!((f.nobs, f.objective), (n, Objective::MaximumLikelihood));

    // Unfittable: the aliased column is also a random slope.
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let cases = [
        (Family::Gaussian, &y, None, 1, Objective::Reml),
        (
            Family::Gaussian,
            &y,
            Some(false),
            1,
            Objective::MaximumLikelihood,
        ),
        (binomial, &yb, None, 1, Objective::Laplace),
        (binomial, &yb, None, 3, Objective::Agq { nodes: 3 }),
    ];
    for (family, y, reml, nagq, want) in cases {
        let f = fit_cold(&x, y, n, p, &spec(family, vec![2]), &ids, &opts(reml, nagq));
        assert!(!f.converged(), "{family:?}, nagq {nagq}");
        assert_eq!(
            (f.nobs, f.objective),
            (n, want),
            "{family:?}, reml {reml:?}, nagq {nagq}"
        );
    }
}

/// The in-crate band between two crate paths: the fit's conditional
/// variances run through the standardized kernel, the reference below
/// through the caller's design.
#[cfg(feature = "formula")]
const CONDVAR_ROUNDOFF: f64 = 1e-10;

/// What [`dense_joint`] returns: `Z` (`n × K`, `K` the length of
/// `Fit::ranef`), `cv` and `kx` in the same column order, and each grouping's
/// `(levels, q)` in declaration order.
#[cfg(feature = "formula")]
pub(crate) struct DenseJoint {
    pub(crate) z: faer::Mat<f64>,
    pub(crate) cv: faer::Mat<f64>,
    pub(crate) kx: faer::Mat<f64>,
    pub(crate) groups: Vec<(usize, usize)>,
}

/// The whole random-effect vector's conditional covariance by brute force on
/// `lo`'s design: `Z` dense over every grouping's reported columns
/// (level-major, the layout of `Fit::ranef`), `Λ` block-diagonal from
/// `varcorr`, `W` the prior weights on a Gaussian fit and the observed
/// curvature at `η̂ = Xβ̂ + Zb̂ + o` otherwise, then `cv = σ̂²·Λ(ΛᵀZᵀWZΛ + I)⁻¹Λᵀ` and
/// `kx = Λ(ΛᵀZᵀWZΛ + I)⁻¹ΛᵀZᵀWX` by dense LU.
#[cfg(feature = "formula")]
pub(crate) fn dense_joint(lo: &crate::formula::Lowered, fit: &Fit) -> DenseJoint {
    use faer::linalg::solvers::Solve;
    use faer::Mat;
    let (n, p, re_p) = (lo.n, lo.p, lo.re_p);
    let spec = lo.spec();
    let re = spec.re.as_ref().unwrap();
    let col = |i: usize, c: usize| -> f64 {
        if c < p {
            lo.x[i * p + c]
        } else {
            lo.re_columns[i * re_p + c - p]
        }
    };
    // Per grouping: row ids, levels, reported columns (`None` the intercept).
    type Group<'a> = (&'a [u32], usize, Vec<Option<usize>>);
    let mut groups: Vec<Group> = Vec::new();
    let reported = |slopes: &[u32], terms: &[crate::ReTerm]| -> Vec<Option<usize>> {
        let hidden = !terms.is_empty() && !terms.iter().any(|t| t.intercept);
        let mut v: Vec<Option<usize>> = if hidden { vec![] } else { vec![None] };
        v.extend(slopes.iter().map(|&c| Some(c as usize)));
        v
    };
    groups.push((
        &lo.ids.primary,
        fit.ranef_levels[0],
        reported(&re.slopes, &re.terms),
    ));
    for (e, gr) in re.extra_groupings.iter().enumerate() {
        groups.push((
            &lo.ids.extra[e],
            fit.ranef_levels[e + 1],
            reported(&gr.slopes, &gr.terms),
        ));
    }
    let gaussian = matches!(spec.family, Family::Gaussian);
    let sigma_sq = if gaussian { fit.dispersion } else { 1.0 };
    // Λ_g: the lower factor of varcorr_g/σ̂², a non-positive pivot a zero column.
    let lams: Vec<Vec<f64>> = groups
        .iter()
        .enumerate()
        .map(|(g, (_, _, cols))| {
            let q = cols.len();
            let mut d = vec![0.0; q * q];
            let mut t = 0;
            for c in 0..q {
                for r in c..q {
                    d[r * q + c] = fit.varcorr[g][t] / sigma_sq;
                    d[c * q + r] = d[r * q + c];
                    t += 1;
                }
            }
            let mut l = vec![0.0f64; q * q];
            for c in 0..q {
                let piv = d[c * q + c] - (0..c).map(|k| l[c * q + k].powi(2)).sum::<f64>();
                if piv <= 1e-14 * d[c * q + c].abs().max(1e-300) {
                    continue;
                }
                l[c * q + c] = piv.sqrt();
                for r in (c + 1)..q {
                    let s = d[r * q + c] - (0..c).map(|k| l[r * q + k] * l[c * q + k]).sum::<f64>();
                    l[r * q + c] = s / l[c * q + c];
                }
            }
            l
        })
        .collect();
    let offs: Vec<usize> = groups
        .iter()
        .scan(0, |acc, (_, lv, cols)| {
            let o = *acc;
            *acc += lv * cols.len();
            Some(o)
        })
        .collect();
    let kt: usize = groups.iter().map(|(_, lv, cols)| lv * cols.len()).sum();
    assert_eq!(kt, fit.ranef.len(), "reference Z width against ranef");
    let z = Mat::<f64>::from_fn(n, kt, |i, k| {
        let g = offs.iter().rposition(|&o| o <= k).unwrap();
        let q = groups[g].2.len();
        let (lvl, r) = ((k - offs[g]) / q, (k - offs[g]) % q);
        if groups[g].0[i] as usize != lvl {
            return 0.0;
        }
        groups[g].2[r].map_or(1.0, |c| col(i, c))
    });
    let lam = Mat::<f64>::from_fn(kt, kt, |a, b| {
        let g = offs.iter().rposition(|&o| o <= a).unwrap();
        let q = groups[g].2.len();
        if b < offs[g] || (b - offs[g]) / q != (a - offs[g]) / q {
            return 0.0;
        }
        lams[g][((a - offs[g]) % q) * q + (b - offs[g]) % q]
    });
    let w: Vec<f64> = (0..n)
        .map(|i| {
            let pw = lo.weights.as_ref().map_or(1.0, |w| w[i]);
            if gaussian {
                return pw;
            }
            let mut eta = lo.offset.as_ref().map_or(0.0, |o| o[i]);
            eta += (0..p).map(|j| lo.x[i * p + j] * fit.beta[j]).sum::<f64>();
            eta += (0..kt).map(|k| z[(i, k)] * fit.ranef[k]).sum::<f64>();
            // The observed curvature, as the GLMM kernel's `W`; it is the
            // Fisher weight on a canonical link.
            let (mu, w, _) =
                crate::family::irls_weight_and_resid(spec.family, f64::NAN, lo.y[i], eta);
            let eta = crate::family::clamp_eta(spec.family, eta);
            crate::family::observed_weight(spec.family, f64::NAN, lo.y[i], pw, eta, mu, pw * w)
        })
        .collect();
    let m = &z * &lam;
    let wm = Mat::<f64>::from_fn(n, kt, |i, k| w[i] * m[(i, k)]);
    let a = m.transpose() * &wm + Mat::<f64>::identity(kt, kt);
    let x = Mat::<f64>::from_fn(n, p, |i, j| lo.x[i * p + j]);
    let ainv = a.partial_piv_lu().solve(Mat::<f64>::identity(kt, kt));
    let cv = (&lam * &ainv * lam.transpose()) * faer::Scale(sigma_sq);
    let kx = &lam * &ainv * wm.transpose() * &x;
    let groups = groups
        .iter()
        .map(|(_, lv, cols)| (*lv, cols.len()))
        .collect();
    DenseJoint { z, cv, kx, groups }
}

/// [`Fit::condvar`] and [`Fit::condvar_x`] read off [`dense_joint`] level by
/// level.
#[cfg(feature = "formula")]
fn dense_condvar(lo: &crate::formula::Lowered, fit: &Fit) -> (Vec<f64>, Vec<f64>) {
    let DenseJoint { cv, kx, groups, .. } = dense_joint(lo, fit);
    let p = lo.p;
    let (mut condvar, mut condvar_x) = (Vec::new(), Vec::new());
    let mut b = 0;
    for (levels, q) in groups {
        for _ in 0..levels {
            for c in 0..q {
                for r in c..q {
                    condvar.push(cv[(b + r, b + c)]);
                }
            }
            for r in 0..q {
                condvar_x.extend((0..p).map(|j| kx[(b + r, j)]));
            }
            b += q;
        }
    }
    (condvar, condvar_x)
}

/// `got` against `want` entry by entry within [`CONDVAR_ROUNDOFF`] relative
/// to `1 + max |want|`; returns the largest relative gap.
#[cfg(feature = "formula")]
fn condvar_gap(got: &[f64], want: &[f64], what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = 1.0 + want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let gap = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs() / scale));
    assert!(
        gap <= CONDVAR_ROUNDOFF,
        "{what}: gap {gap:e} over {CONDVAR_ROUNDOFF:e}"
    );
    gap
}

/// `Fit::condvar` and `Fit::condvar_x` equal a dense inverse of the
/// mixed-model matrix at the fitted parameters on every covered layout: the
/// per-group, nested and dense crossed LMM routes, the hidden column of an
/// intercept-free block and an uncorrelated block, a random-effect-only slope,
/// and the GLMM blocked and structured layouts (the latter with the grouping
/// order the size rule reverses). The default options form none, and leave
/// every other estimate bitwise where the `condvar: true` fit has it.
#[cfg(feature = "formula")]
#[test]
fn condvar_matches_a_dense_inverse() {
    let sleep = include_str!("../../validation/data/empirical/sleepstudy.csv");
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let shapes: [(&str, &[&str], &str, Family); 8] = [
        (
            sleep,
            &["Subject"],
            "Reaction ~ Days + (1 + Days | Subject)",
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
            sleep,
            &["Subject"],
            "Reaction ~ Days + (Days || Subject)",
            Family::Gaussian,
        ),
        (
            sleep,
            &["Subject"],
            "Reaction ~ Days + (0 + Days | Subject)",
            Family::Gaussian,
        ),
        (
            sleep,
            &["Subject"],
            "Reaction ~ 1 + (Days | Subject)",
            Family::Gaussian,
        ),
        (
            include_str!("../../validation/data/simulated/sim_factorial_bern.csv"),
            &["subj", "g", "h"],
            "y ~ x + g + (1 | subj)",
            binomial,
        ),
        (
            include_str!("../../validation/data/empirical/grouseticks.csv"),
            &["INDEX", "BROOD", "YEAR", "LOCATION"],
            "TICKS ~ YEAR + cHEIGHT + (1 | LOCATION) + (1 | BROOD)",
            poisson,
        ),
    ];
    let mut ran = 0;
    for (csv, factors, formula, family) in shapes {
        let lo = fixture_lowered(csv, factors, formula, family);
        let opts = FitOptions {
            condvar: true,
            ..FitOptions::default()
        };
        let fit = crate::fit(&lo.data(), lo.spec(), &opts).unwrap();
        assert!(fit.converged(), "{formula}: not converged");
        assert!(
            !fit.condvar.is_empty(),
            "{formula}: no conditional variances"
        );
        let off = crate::fit(&lo.data(), lo.spec(), &FitOptions::default()).unwrap();
        assert!(off.condvar.is_empty() && off.condvar_x.is_empty() && off.condvar_cross.is_none());
        let bits = |v: &[f64]| v.iter().map(|b| b.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&off.beta), bits(&fit.beta), "{formula}: beta moved");
        assert_eq!(bits(&off.se), bits(&fit.se), "{formula}: se moved");
        assert_eq!(bits(&off.ranef), bits(&fit.ranef), "{formula}: ranef moved");
        assert_eq!(
            off.deviance.to_bits(),
            fit.deviance.to_bits(),
            "{formula}: deviance moved"
        );
        let (cv, cx) = dense_condvar(&lo, &fit);
        // `label_ranef` keeps a labelled level's vech and drops a padded slot's.
        let blocks = crate::formula::label_ranef(&fit, &lo.re_groups).unwrap();
        let mut at = 0;
        for (block, info) in blocks.iter().zip(&lo.re_groups) {
            let v = info.terms.len() * (info.terms.len() + 1) / 2;
            let kept: Vec<f64> = (0..info.slot_labels.len())
                .filter(|&l| info.slot_labels[l].is_some())
                .flat_map(|l| fit.condvar[at + l * v..at + (l + 1) * v].to_vec())
                .collect();
            assert_eq!(block.condvar, kept, "{formula}: RanefBlock::condvar");
            assert_eq!(block.condvar.len(), block.levels.len() * v);
            at += info.slot_labels.len() * v;
        }
        let g1 = condvar_gap(&fit.condvar, &cv, &format!("{formula} condvar"));
        let g2 = condvar_gap(&fit.condvar_x, &cx, &format!("{formula} condvar_x"));
        eprintln!("{formula}: condvar {g1:e}, condvar_x {g2:e}");
        ran += 1;
    }
    assert_eq!(ran, 8);
}

/// Where the conditional variances are not computed, `condvar` and
/// `condvar_x` are empty while `ranef` is not: a Gaussian LMM on the sparse
/// route (a random slope on an extra grouping) and a GLMM on the packed layout
/// (the same shape, binomial).
#[cfg(feature = "formula")]
#[test]
fn condvar_is_empty_where_not_covered() {
    let shapes: [(&str, &[&str], &str, Family); 2] = [
        (
            include_str!("../../validation/data/simulated/sim_slope_extra.csv"),
            &["g1", "g2"],
            "y ~ 1 + x + (1 | g1) + (1 + x | g2)",
            Family::Gaussian,
        ),
        (
            include_str!("../../validation/data/simulated/sim_factorial_bern.csv"),
            &["subj", "g", "h"],
            "y ~ x + (1 | subj) + (1 + x | g)",
            Family::Binomial {
                link: BinomialLink::Logit,
            },
        ),
    ];
    for (csv, factors, formula, family) in shapes {
        let lo = fixture_lowered(csv, factors, formula, family);
        let fit = crate::fit(&lo.data(), lo.spec(), &FitOptions::default()).unwrap();
        assert!(!fit.ranef.is_empty(), "{formula}: no ranef");
        assert!(fit.condvar.is_empty(), "{formula}: condvar");
        assert!(fit.condvar_x.is_empty(), "{formula}: condvar_x");
    }
}
