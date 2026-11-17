#![cfg(feature = "formula")]
//! Every fit state the kernel reaches (converged, singular, failed, aliased),
//! on every model shape where it exists, through every public call on a fit.
//! A call must return or refuse with its own error type, never panic; on a fit
//! with no estimates (every β NaN) every estimate it returns is NaN; and every
//! per-grouping output comes in the order the formula declares the groupings.
//!
//! The data table is deterministic (no RNG) and is built the same way in
//! `glmm-py/tests/test_fit_states.py` and
//! `glmm-r/tests/testthat/test-fit-states.R`: change the three together.

use std::panic::{catch_unwind, AssertUnwindSafe};

use glmm::formula::anova::{compare, term_tests, CompareOptions, TermTestOptions};
use glmm::formula::{self, pairwise, Adjust, Column, Error, Model, Table};
use glmm::inference::{ProfileParam, Reference};
use glmm::{
    BinomialLink, Family, FitError, FitOptions, Note, PredictOptions, Scale, SeType, Simulator,
};

const N: usize = 96;

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    Converged,
    Singular,
    ConstantResponse,
    Separated,
    TooFewRows,
    AliasedFixed,
    AliasedSlope,
}

/// A `y ~ f + x`; B `+ (1 | g)`; C `+ (1 | a) + (1 | b)`; D `+ (1 + x | g)`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Shape {
    A,
    B,
    C,
    D,
}

#[derive(Clone, Copy, Debug)]
struct Row {
    state: State,
    binomial: bool,
    shape: Shape,
}

impl Row {
    fn label(&self) -> String {
        let family = if self.binomial {
            "binomial"
        } else {
            "gaussian"
        };
        format!("{:?}/{family}/{:?}", self.state, self.shape)
    }

    fn family(&self) -> Family {
        if self.binomial {
            Family::Binomial {
                link: BinomialLink::Logit,
            }
        } else {
            Family::Gaussian
        }
    }

    fn formula(&self) -> String {
        let re = match self.shape {
            Shape::A => "",
            Shape::B => " + (1 | g)",
            Shape::C => " + (1 | a) + (1 | b)",
            Shape::D => " + (1 + x | g)",
        };
        match self.state {
            State::AliasedFixed => format!("y ~ f + x + x2{re}"),
            State::AliasedSlope => "y ~ f + x + x2 + (1 + x2 | g)".to_string(),
            _ => format!("y ~ f + x{re}"),
        }
    }

    /// The row's formula without `x` in the fixed part, for the two-fit calls.
    fn reduced_formula(&self) -> String {
        self.formula().replacen(" + x", "", 1)
    }

    /// The declared groupings and the random effects each one carries.
    fn groupings(&self) -> Vec<(&'static str, usize)> {
        match (self.state, self.shape) {
            (State::AliasedSlope, _) | (_, Shape::D) => vec![("g", 2)],
            (_, Shape::A) => vec![],
            (_, Shape::B) => vec![("g", 1)],
            (_, Shape::C) => vec![("a", 1), ("b", 1)],
        }
    }
}

/// The 39 fits: every (state, family, shape) that exists.
fn rows() -> Vec<Row> {
    use Shape::*;
    use State::*;
    let mut rows = Vec::new();
    let mut push = |state, binomial, shapes: &[Shape]| {
        for &shape in shapes {
            rows.push(Row {
                state,
                binomial,
                shape,
            });
        }
    };
    for binomial in [false, true] {
        push(Converged, binomial, &[A, B, C, D]);
        push(Singular, binomial, &[B, C, D]);
        push(TooFewRows, binomial, &[A, B, C, D]);
        push(AliasedFixed, binomial, &[A, B, C, D]);
    }
    push(ConstantResponse, false, &[A, B, C, D]);
    push(Separated, true, &[A, B, C, D]);
    push(AliasedSlope, false, &[D]);
    rows
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

fn mean_by(values: &[f64], key: &[usize]) -> Vec<f64> {
    let k = key.iter().max().map_or(0, |m| m + 1);
    let (mut sum, mut count) = (vec![0.0; k], vec![0.0; k]);
    for (v, &j) in values.iter().zip(key) {
        sum[j] += v;
        count[j] += 1.0;
    }
    sum.iter().zip(&count).map(|(s, c)| s / c).collect()
}

/// The response of `row` on rows `0..N`.
fn response(row: &Row) -> Vec<f64> {
    let idx: Vec<usize> = (0..N).collect();
    let x: Vec<f64> = idx.iter().map(|&i| (1.7 * i as f64 + 0.3).sin()).collect();
    let z: Vec<f64> = idx
        .iter()
        .map(|&i| 0.8 * (2.3 * i as f64 + 0.7).sin())
        .collect();
    let g: Vec<usize> = idx.iter().map(|&i| (i / 3) % 6).collect();
    let a: Vec<usize> = idx.iter().map(|&i| i % 4).collect();
    let b: Vec<usize> = idx.iter().map(|&i| (i / 4) % 12).collect();
    match (row.state, row.binomial) {
        (State::ConstantResponse, _) => vec![3.0; N],
        (State::Separated, _) => x.iter().map(|&v| f64::from(u8::from(v > 0.0))).collect(),
        (State::Singular, true) => idx
            .iter()
            .map(|&i| f64::from(u8::from((i / 16) % 2 == 0)))
            .collect(),
        (State::Singular, false) => {
            if row.shape == Shape::C {
                let (ma, mb) = (mean_by(&z, &a), mean_by(&z, &b));
                let grand = z.iter().sum::<f64>() / N as f64;
                (0..N).map(|i| z[i] - ma[a[i]] - mb[b[i]] + grand).collect()
            } else {
                let mg = mean_by(&z, &g);
                (0..N).map(|i| z[i] - mg[g[i]]).collect()
            }
        }
        _ => {
            let u = [-1.0, 0.6, 0.2, -0.4, 1.1, -0.5];
            let s = [0.5, -0.4, 0.3, -0.6, 0.2, 0.4];
            let ua = [-0.8, 0.5, 0.9, -0.6];
            let shape = if row.state == State::AliasedSlope {
                Shape::D
            } else {
                row.shape
            };
            let eta: Vec<f64> = (0..N)
                .map(|i| {
                    let fe = 0.5 * x[i] + [0.0, 0.3, -0.4][i % 3];
                    fe + match shape {
                        Shape::A => 0.0,
                        Shape::B => u[g[i]],
                        Shape::C => ua[a[i]] + 0.7 * (b[i] as f64 + 1.0).sin(),
                        Shape::D => u[g[i]] + s[g[i]] * x[i],
                    }
                })
                .collect();
            if row.binomial {
                (0..N)
                    .map(|i| {
                        let t = frac(0.618034 * i as f64 + 0.1);
                        f64::from(u8::from(t < 1.0 / (1.0 + (-2.0 * eta[i]).exp())))
                    })
                    .collect()
            } else {
                (0..N).map(|i| 1.0 + eta[i] + z[i]).collect()
            }
        }
    }
}

/// The table `row` is fitted on: all `N` rows, or rows 0, 13, 26 and 39 of
/// them for a too-few-rows fit (n = p = 4).
fn table(row: &Row) -> Table {
    let keep: Vec<usize> = if row.state == State::TooFewRows {
        vec![0, 13, 26, 39]
    } else {
        (0..N).collect()
    };
    let y = response(row);
    let labels =
        |f: &dyn Fn(usize) -> String| -> Vec<String> { keep.iter().map(|&i| f(i)).collect() };
    let x: Vec<f64> = keep.iter().map(|&i| (1.7 * i as f64 + 0.3).sin()).collect();
    Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(keep.iter().map(|&i| y[i]).collect()),
            ),
            (
                "f".into(),
                Column::factor_from_labels(&labels(&|i| ["p", "q", "r"][i % 3].to_string())),
            ),
            (
                "g".into(),
                Column::factor_from_labels(&labels(&|i| format!("g{}", (i / 3) % 6))),
            ),
            (
                "a".into(),
                Column::factor_from_labels(&labels(&|i| format!("a{}", i % 4))),
            ),
            (
                "b".into(),
                Column::factor_from_labels(&labels(&|i| format!("b{}", (i / 4) % 12))),
            ),
            (
                "x2".into(),
                Column::Numeric(x.iter().map(|v| 2.0 * v).collect()),
            ),
            ("x".into(), Column::Numeric(x)),
        ],
        n: keep.len(),
    }
}

/// The ports' fit options: `condvar` on, so the conditional standard errors
/// of `predict` are reached.
fn options() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.condvar = true;
    opts
}

/// One fitted row and what its calls need.
struct Ctx {
    row: Row,
    model: Model,
    /// The fit's own table, the new rows of `predict`.
    table: Table,
    /// The reduced model of the two-fit calls, or the reason it did not fit.
    reduced: Result<Model, String>,
}

impl Ctx {
    fn new(row: Row) -> Ctx {
        let table = table(&row);
        let model = formula::fit(&row.formula(), &table, row.family(), &options())
            .unwrap_or_else(|e| panic!("{}: the fit was refused: {e}", row.label()));
        let reduced = formula::fit(&row.reduced_formula(), &table, row.family(), &options())
            .map_err(|e| e.to_string());
        Ctx {
            row,
            model,
            table,
            reduced,
        }
    }

    fn fit(&self) -> &glmm::Fit {
        self.model.fit()
    }

    /// No estimates: every β NaN.
    fn no_estimates(&self) -> bool {
        self.fit().beta().iter().all(|b| b.is_nan())
    }

    /// Unit row `j` of length p.
    fn unit(&self, j: usize) -> Vec<f64> {
        let mut l = vec![0.0; self.fit().beta().len()];
        l[j] = 1.0;
        l
    }

    /// `per(g)` over every grouping, concatenated.
    fn per_grouping(&self, per: impl Fn(usize) -> Vec<f64>) -> Vec<f64> {
        (0..self.fit().n_groupings()).flat_map(per).collect()
    }

    fn predict(&self, scale: Scale) -> Result<Vec<f64>, String> {
        let mut opts = PredictOptions::default();
        opts.se = true;
        opts.scale = scale;
        let p = formula::predict(&self.model, &self.table, &opts).map_err(|e| e.to_string())?;
        Ok([p.fit, p.se, p.lower, p.upper].concat())
    }
}

fn fits() -> Vec<Ctx> {
    rows().into_iter().map(Ctx::new).collect()
}

type Call = fn(&Ctx) -> Result<Vec<f64>, String>;

/// Every public call on a fit: its name, whether its numbers are estimates
/// (NaN on a fit with no estimates), and the call with its fixed arguments.
/// A name without `::` is a `pub fn` of `impl Fit`, the list
/// `every_pub_fn_on_fit_is_called_or_skipped` checks; the others are the
/// calls of the formula door and the simulator that read a fit.
const CALLS: &[(&str, bool, Call)] = &[
    ("converged", false, |c| {
        Ok(vec![f64::from(u8::from(c.fit().converged()))])
    }),
    ("singular", false, |c| {
        Ok(vec![f64::from(u8::from(c.fit().singular()))])
    }),
    ("aliased", false, |c| {
        Ok(c.fit()
            .aliased()
            .iter()
            .map(|&a| f64::from(u8::from(a)))
            .collect())
    }),
    ("beta", true, |c| Ok(c.fit().beta().to_vec())),
    ("se", true, |c| Ok(c.fit().se().to_vec())),
    ("vcov", true, |c| Ok(c.fit().vcov().to_vec())),
    ("n_groupings", false, |c| {
        Ok(vec![c.fit().n_groupings() as f64])
    }),
    ("n_re", false, |c| {
        Ok(c.per_grouping(|g| vec![c.fit().n_re(g) as f64]))
    }),
    ("varcorr", true, |c| {
        Ok(c.per_grouping(|g| c.fit().varcorr(g)))
    }),
    ("stddev_se", true, |c| {
        Ok(c.per_grouping(|g| c.fit().stddev_se(g)))
    }),
    ("ranef", true, |c| {
        Ok(c.per_grouping(|g| c.fit().ranef(g).to_vec()))
    }),
    ("ranef_levels", false, |c| {
        Ok(c.fit().ranef_levels().iter().map(|&l| l as f64).collect())
    }),
    ("condvar", true, |c| {
        Ok(c.per_grouping(|g| c.fit().condvar(g).to_vec()))
    }),
    ("coef_df", true, |c| Ok(c.fit().coef_df().to_vec())),
    ("fitted", true, |c| Ok(c.fit().fitted().to_vec())),
    ("loglik", true, |c| Ok(vec![c.fit().loglik()])),
    ("deviance", true, |c| Ok(vec![c.fit().deviance()])),
    ("df", false, |c| Ok(vec![c.fit().df() as f64])),
    ("nobs", false, |c| Ok(vec![c.fit().nobs() as f64])),
    ("dispersion", true, |c| Ok(vec![c.fit().dispersion()])),
    ("reml", false, |c| {
        Ok(vec![f64::from(u8::from(c.fit().reml()))])
    }),
    ("weight_power", true, |c| Ok(vec![c.fit().weight_power()])),
    ("se_type", false, |c| {
        Ok(vec![f64::from(u8::from(
            c.fit().se_type() == SeType::Robust,
        ))])
    }),
    ("objective", false, |c| {
        let _ = c.fit().objective();
        Ok(vec![])
    }),
    ("n_eval", false, |c| Ok(vec![c.fit().n_eval() as f64])),
    ("diagnostics", false, |c| {
        Ok(vec![c.fit().diagnostics().notes.len() as f64])
    }),
    // The diagonal of the correlation matrix is 1 by definition, not an
    // estimate, so only the SDs and the off-diagonal correlations are returned.
    ("stddev_corr", true, |c| {
        Ok(c.per_grouping(|g| {
            let (sd, corr) = c.fit().stddev_corr(g);
            let off = (0..sd.len()).flat_map(|r| (0..r).map(move |k| (r, k)));
            sd.iter()
                .copied()
                .chain(off.map(|(r, k)| corr[r][k]))
                .collect()
        }))
    }),
    ("contrast_t", true, |c| {
        let t = c.fit().contrast_t(&c.unit(0)).map_err(|e| e.to_string())?;
        Ok(vec![t.estimate, t.se, t.t, t.df, t.p_value])
    }),
    ("contrast_f", true, |c| {
        let p = c.fit().beta().len();
        let l = [c.unit(0), c.unit(p - 1)].concat();
        let f = c.fit().contrast_f(&l, 2).map_err(|e| e.to_string())?;
        Ok(vec![f.f, f.den_df, f.p_value])
    }),
    ("r2", true, |c| {
        let lo = c.model.lowered();
        let r = c
            .fit()
            .r2(&lo.data(), lo.spec())
            .map_err(|e| e.to_string())?;
        Ok(vec![r.marginal, r.conditional])
    }),
    ("icc", true, |c| {
        let lo = c.model.lowered();
        let r = c
            .fit()
            .icc(&lo.data(), lo.spec())
            .map_err(|e| e.to_string())?;
        Ok(vec![r.adjusted, r.unadjusted])
    }),
    ("profile_intervals", true, |c| {
        let lo = c.model.lowered();
        let p = c
            .fit()
            .profile_intervals(&lo.data(), lo.spec(), 0.95, None)
            .map_err(|e| e.to_string())?;
        let rows = p.rows.iter().flat_map(|r| [r.estimate, r.lower, r.upper]);
        Ok(std::iter::once(p.deviance).chain(rows).collect())
    }),
    ("formula::predict(link)", true, |c| c.predict(Scale::Link)),
    ("formula::predict(response)", true, |c| {
        c.predict(Scale::Response)
    }),
    ("Simulator::simulate", true, |c| {
        let lo = c.model.lowered();
        let design = lo.design();
        let sim = Simulator::new(c.fit(), lo.spec(), &design).map_err(|e| e.to_string())?;
        Ok(sim.simulate(1, 0))
    }),
    ("anova::term_tests", true, |c| {
        let t = term_tests(&c.model, &TermTestOptions::default()).map_err(|e| e.to_string())?;
        Ok(t.fixed
            .iter()
            .flat_map(|r| {
                let den = match r.test.reference {
                    Reference::F { den_df, .. } => Some(den_df),
                    _ => None,
                };
                [
                    Some(r.test.statistic),
                    Some(r.test.p_value),
                    r.sum_sq,
                    r.mean_sq,
                    den,
                ]
            })
            .flatten()
            .collect())
    }),
    // Only this fit's own numbers and the test: the reduced model is another
    // fit, which can have estimates where this one has none.
    ("anova::compare", true, |c| {
        let reduced = c.reduced.as_ref().map_err(Clone::clone)?;
        let opts = CompareOptions {
            nsim: 1000,
            seed: 1,
        };
        let cmp = compare(&c.model, reduced, &opts).map_err(|e| e.to_string())?;
        let own = if cmp.m0.formula_label == c.model.formula() {
            &cmp.m0
        } else {
            &cmp.m1
        };
        let (statistic, p_value) = match &cmp.test {
            glmm::formula::anova::ComparisonTest::Lr(t) => (t.statistic, t.p_value),
            glmm::formula::anova::ComparisonTest::Linear(t) => (t.statistic, t.p_value),
            _ => (f64::NAN, f64::NAN),
        };
        let ic = [own.aic, own.bic].into_iter().flatten();
        Ok([own.loglik, statistic, p_value]
            .into_iter()
            .chain(ic)
            .collect())
    }),
    ("formula::pairwise", true, |c| {
        let pw = pairwise(&c.model, "f", None, Adjust::Holm).map_err(|e| e.to_string())?;
        Ok(pw
            .rows
            .iter()
            // An infinite df marks a z row, a label of the test, not an estimate.
            .flat_map(|r| {
                let df = (r.df != f64::INFINITY).then_some(r.df);
                [r.estimate, r.se, r.statistic, r.p_value, r.p_adjusted]
                    .into_iter()
                    .chain(df)
            })
            .collect())
    }),
    ("Pairwise::confint", true, |c| {
        let pw = pairwise(&c.model, "f", None, Adjust::Holm).map_err(|e| e.to_string())?;
        let ci = pw.confint(0.95).map_err(|e| e.to_string())?;
        Ok(ci.rows.iter().flat_map(|r| [r.lower, r.upper]).collect())
    }),
    ("Model::ranef", true, |c| {
        let blocks = c.model.ranef().map_err(|e| e.to_string())?;
        Ok(blocks
            .into_iter()
            .flat_map(|b| [b.values, b.condvar].concat())
            .collect())
    }),
];

/// `pub fn`s of `impl Fit` no row calls, with the reason.
const SKIP: &[(&str, &str)] = &[(
    "counters",
    "only with the `counters` feature: an instrumentation tally, not a fit output",
)];

/// Runs `call` on `ctx`, a panic caught and returned as its message.
fn run(call: Call, ctx: &Ctx) -> Result<Result<Vec<f64>, String>, String> {
    catch_unwind(AssertUnwindSafe(|| call(ctx))).map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    })
}

#[test]
fn every_fit_state_reaches_its_state() {
    let mut failures = Vec::new();
    let all = rows();
    for row in &all {
        let ctx = Ctx::new(*row);
        let fit = ctx.fit();
        let separated = fit
            .diagnostics()
            .notes
            .iter()
            .any(|n| matches!(n, Note::Separated));
        let x2 = ctx.model.lowered().col_names.iter().position(|n| n == "x2");
        let x2_aliased = x2.is_some_and(|j| fit.aliased()[j] && fit.beta()[j].is_nan());
        let ok = match row.state {
            State::Converged => fit.converged() && !fit.singular(),
            State::Singular => fit.converged() && fit.singular(),
            State::ConstantResponse if row.shape == Shape::A => fit.converged(),
            State::ConstantResponse | State::TooFewRows => !fit.converged() && ctx.no_estimates(),
            State::Separated => !fit.converged() && separated,
            State::AliasedFixed | State::AliasedSlope => fit.converged() && x2_aliased,
        };
        if !ok {
            failures.push(format!(
                "{}: converged {}, singular {}, β {:?}, aliased {:?}, notes {:?}",
                row.label(),
                fit.converged(),
                fit.singular(),
                fit.beta(),
                fit.aliased(),
                fit.diagnostics().notes
            ));
        }
    }
    assert_eq!(all.len(), 39);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_call_on_every_fit_returns_or_refuses() {
    let fits = fits();
    let mut failures = Vec::new();
    let mut ran = 0;
    for ctx in &fits {
        for &(name, estimate, call) in CALLS {
            ran += 1;
            let at = format!("{}: {name}", ctx.row.label());
            let values = match run(call, ctx) {
                Err(panic) => {
                    failures.push(format!("{at}: panicked: {panic}"));
                    continue;
                }
                Ok(Err(_refusal)) => continue,
                Ok(Ok(values)) => values,
            };
            if estimate && ctx.no_estimates() && values.iter().any(|v| !v.is_nan()) {
                failures.push(format!(
                    "{at}: a number is not NaN on a fit with no estimates: {values:?}"
                ));
            }
            if name == "Simulator::simulate"
                && ctx.fit().converged()
                && values.iter().any(|v| !v.is_finite())
            {
                failures.push(format!(
                    "{at}: a draw is not finite on a converged fit: {values:?}"
                ));
            }
        }
    }
    assert_eq!((fits.len(), ran), (39, 39 * CALLS.len()));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn per_grouping_outputs_follow_the_declaration_order() {
    let fits = fits();
    let mut failures = Vec::new();
    let mut checked = 0;
    for ctx in &fits {
        let declared = ctx.row.groupings();
        if declared.is_empty() {
            continue;
        }
        checked += 1;
        let names: Vec<&str> = declared.iter().map(|d| d.0).collect();
        // Level counts from the table itself: a too-few-rows table keeps 4 of each.
        let levels: Vec<usize> = names
            .iter()
            .map(
                |name| match ctx.table.columns.iter().find(|c| c.0 == *name) {
                    Some((_, Column::Factor { codes, .. })) => {
                        let mut c = codes.clone();
                        c.sort_unstable();
                        c.dedup();
                        c.len()
                    }
                    _ => unreachable!("{name} is a factor column"),
                },
            )
            .collect();
        let check = || -> Vec<String> {
            let fit = ctx.fit();
            let mut bad = Vec::new();
            let mut expect = |what: &str, ok: bool, got: String| {
                if !ok {
                    bad.push(format!("{what}: {got}"));
                }
            };
            expect(
                "n_groupings",
                fit.n_groupings() == names.len(),
                format!("{}", fit.n_groupings()),
            );
            expect(
                "ranef_levels",
                fit.ranef_levels() == levels,
                format!("{:?} for {levels:?}", fit.ranef_levels()),
            );
            for (g, &(_, q)) in declared.iter().enumerate().take(fit.n_groupings()) {
                let lv = fit.ranef_levels().get(g).copied().unwrap_or(0);
                expect("n_re", fit.n_re(g) == q, format!("{} at {g}", fit.n_re(g)));
                expect(
                    "varcorr",
                    fit.varcorr(g).len() == q * q,
                    format!("{} at {g}", fit.varcorr(g).len()),
                );
                expect(
                    "ranef",
                    fit.ranef(g).len() == lv * q,
                    format!("{} at {g}", fit.ranef(g).len()),
                );
                let cv = fit.condvar(g).len();
                expect(
                    "condvar",
                    cv == 0 || cv == lv * q * (q + 1) / 2,
                    format!("{cv} at {g}"),
                );
            }
            match ctx.model.ranef() {
                Ok(blocks) if !blocks.is_empty() => {
                    let got: Vec<(&str, usize)> = blocks
                        .iter()
                        .map(|b| (b.group.as_str(), b.levels.len()))
                        .collect();
                    let want: Vec<(&str, usize)> =
                        names.iter().copied().zip(levels.iter().copied()).collect();
                    expect("Model::ranef", got == want, format!("{got:?} for {want:?}"));
                }
                Ok(_) => {}
                Err(e) => expect("Model::ranef", false, e.to_string()),
            }
            let lo = ctx.model.lowered();
            if let Ok(p) = fit.profile_intervals(&lo.data(), lo.spec(), 0.95, None) {
                let groups: Vec<usize> = p
                    .rows
                    .iter()
                    .filter_map(|r| match r.param {
                        ProfileParam::Sd { group, .. } => Some(group),
                        _ => None,
                    })
                    .collect();
                let ordered = groups.windows(2).all(|w| w[0] <= w[1])
                    && groups.iter().all(|&g| g < names.len());
                expect("profile_intervals", ordered, format!("{groups:?}"));
            }
            bad
        };
        match catch_unwind(AssertUnwindSafe(check)) {
            Ok(bad) => {
                failures.extend(bad.into_iter().map(|b| format!("{}: {b}", ctx.row.label())))
            }
            Err(_) => failures.push(format!("{}: panicked", ctx.row.label())),
        }
    }
    assert_eq!(checked, 31);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The names of the `pub fn`s inside every `impl Fit {` block of the crate's
/// sources, read at test time so a block in a new file is found too.
fn pub_fns_on_fit() -> Vec<String> {
    fn walk(dir: &std::path::Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let mut inside = false;
                for line in std::fs::read_to_string(&path).unwrap().lines() {
                    if line == "impl Fit {" {
                        inside = true;
                    } else if line == "}" {
                        inside = false;
                    } else if let Some(rest) = line.strip_prefix("    pub fn ").filter(|_| inside) {
                        let end = rest.find(['(', '<']).unwrap_or(rest.len());
                        found.push(rest[..end].to_string());
                    }
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
        &mut found,
    );
    found
}

#[test]
fn every_pub_fn_on_fit_is_called_or_skipped() {
    use std::collections::BTreeSet;
    let found: BTreeSet<String> = pub_fns_on_fit().into_iter().collect();
    let listed: BTreeSet<String> = CALLS
        .iter()
        .map(|c| c.0)
        .filter(|n| !n.contains("::"))
        .chain(SKIP.iter().map(|s| s.0))
        .map(str::to_string)
        .collect();
    assert!(
        found.len() >= 33,
        "found {} pub fns: {found:?}",
        found.len()
    );
    let missing: Vec<_> = found.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&found).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "not called and not skipped: {missing:?}; listed but not a pub fn of Fit: {stale:?}"
    );
}

#[test]
fn a_fit_refused_before_fitting_is_an_err() {
    let empty = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![])),
            ("x".into(), Column::Numeric(vec![])),
            ("g".into(), Column::factor_from_labels(&[])),
        ],
        n: 0,
    };
    let empty_fit = formula::fit("y ~ x + (1 | g)", &empty, Family::Gaussian, &options());
    assert!(
        matches!(empty_fit, Err(Error::Fit(FitError::Shape(_)))),
        "{:?}",
        empty_fit.err().map(|e| e.to_string())
    );
    let converged = Row {
        state: State::Converged,
        binomial: false,
        shape: Shape::B,
    };
    let no_response = formula::fit("~ x", &table(&converged), Family::Gaussian, &options());
    assert!(
        matches!(no_response, Err(Error::Parse(_))),
        "{:?}",
        no_response.err().map(|e| e.to_string())
    );
}

/// The Gaussian rows through the ports' entry point: the Satterthwaite record
/// is absent, or square in `n_free + 1` with one Jacobian matrix per free θ
/// coordinate, and finite on a converged fit.
#[cfg(feature = "orchestrate")]
#[test]
fn satterthwaite_record_on_every_gaussian_fit() {
    use glmm::orchestrate::run_fit;
    use std::collections::HashMap;
    let mut failures = Vec::new();
    let (mut ran, mut records) = (0, 0);
    for row in rows().into_iter().filter(|r| !r.binomial) {
        ran += 1;
        let (mut numeric, mut factors) = (HashMap::new(), HashMap::new());
        for (name, col) in table(&row).columns {
            match col {
                Column::Numeric(v) => {
                    numeric.insert(name, v);
                }
                Column::Factor { levels, codes } => {
                    factors.insert(name, (levels, codes));
                }
            }
        }
        let fitted = run_fit(
            &row.formula(),
            numeric,
            factors,
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            None,
            None,
            None,
            true,
            true,
            None,
            "Wald",
            None,
            None,
        );
        let Ok(r) = fitted else { continue };
        let Some(rec) = r.satterthwaite_record else {
            continue;
        };
        records += 1;
        let k = (rec.a_eff.len() as f64).sqrt().round() as usize;
        let shaped = k >= 1 && k * k == rec.a_eff.len() && rec.jac_theta.len() == k - 1;
        let numbers = rec
            .a_eff
            .iter()
            .chain(rec.jac_theta.iter().flatten())
            .chain(&rec.c)
            .chain([&rec.tau, &rec.nu]);
        let finite = !r.converged || numbers.into_iter().all(|v| v.is_finite());
        if !shaped || !finite {
            failures.push(format!("{}: {rec:?}", row.label()));
        }
    }
    assert_eq!(ran, 20);
    assert_eq!(records, 10);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Population-mode `predict` of every fit on its own table finds every row
/// estimable: the aliased fits' null-space basis annihilates their training
/// rows.
#[test]
fn predicting_the_training_rows_finds_every_row_estimable() {
    const PREDICTED: usize = 39;
    let mut opts = PredictOptions::default();
    opts.population = true;
    let fits = fits();
    let (mut predicted, mut aliased_predicted) = (0, 0);
    for c in &fits {
        if let Ok(p) = formula::predict(&c.model, &c.table, &opts) {
            predicted += 1;
            if matches!(c.row.state, State::AliasedFixed | State::AliasedSlope) {
                aliased_predicted += 1;
            }
            let lost: Vec<&Note> = p
                .notes
                .iter()
                .filter(|n| matches!(n, Note::RowsNotEstimable { .. }))
                .collect();
            assert!(lost.is_empty(), "{}: {lost:?}", c.row.label());
        }
    }
    assert_eq!(fits.len(), 39);
    assert_eq!(predicted, PREDICTED);
    assert_eq!(aliased_predicted, 9);
}

/// No fit carries `Note::NbShapeAtLimit`: no row is negative binomial, so the
/// note on any of them is raised on the wrong family.
#[test]
fn no_fit_is_at_the_negative_binomial_shape_limit() {
    let fits = fits();
    for c in &fits {
        let notes = &c.fit().diagnostics().notes;
        assert!(
            !notes
                .iter()
                .any(|n| matches!(n, Note::NbShapeAtLimit { .. })),
            "{}: {notes:?}",
            c.row.label()
        );
    }
    assert_eq!(fits.len(), 39);
}

/// No fit carries `Note::DispersionAtLimit`: no row is Gamma, so the note on
/// any of them is raised on the wrong family.
#[test]
fn no_fit_is_at_the_gamma_dispersion_limit() {
    let fits = fits();
    for c in &fits {
        let notes = &c.fit().diagnostics().notes;
        assert!(
            !notes
                .iter()
                .any(|n| matches!(n, Note::DispersionAtLimit { .. })),
            "{}: {notes:?}",
            c.row.label()
        );
    }
    assert_eq!(fits.len(), 39);
}

/// `Note::OneTrialPerLevel` marks exactly the binomial too-few-rows fits with
/// a grouping: their rows 0, 13, 26 and 39 fall in four different levels of
/// `g`, of `a` and of `b`, so every level holds one 0/1 row. No other fit
/// carries it.
#[test]
fn one_trial_per_level_marks_the_one_row_per_level_fits() {
    let fits = fits();
    for c in &fits {
        let got: Vec<u32> = (c.fit().diagnostics().notes.iter())
            .filter_map(|n| match n {
                Note::OneTrialPerLevel { grouping } => Some(*grouping),
                _ => None,
            })
            .collect();
        let want: Vec<u32> = match (c.row.state, c.row.binomial, c.row.shape) {
            (State::TooFewRows, true, Shape::B | Shape::D) => vec![0],
            (State::TooFewRows, true, Shape::C) => vec![0, 1],
            _ => vec![],
        };
        assert_eq!(got, want, "{}", c.row.label());
    }
    assert_eq!(fits.len(), 39);
}

/// `coef_df` of a robust fit is finite and positive, or NaN: on the Gaussian
/// rows and on two nearly collinear designs (`xi2 = xi + 1e-7·w`), HC2 on OLS
/// and CR2 with a random intercept.
#[test]
fn robust_df_is_positive_or_nan() {
    let mut opts = options();
    opts.se_type = Some(SeType::Robust);
    let mut failures = Vec::new();
    let mut check = |label: String, model: Result<Model, Error>| {
        if let Ok(m) = model {
            let df = m.fit().coef_df();
            if df
                .iter()
                .any(|d| !(d.is_nan() || (d.is_finite() && *d > 0.0)))
            {
                failures.push(format!("{label}: coef_df {df:?}"));
            }
        }
    };
    let gaussian: Vec<Row> = rows().into_iter().filter(|r| !r.binomial).collect();
    for row in &gaussian {
        check(
            row.label(),
            formula::fit(&row.formula(), &table(row), Family::Gaussian, &opts),
        );
    }
    let n = 120;
    let col = |f: &dyn Fn(f64) -> f64| -> Vec<f64> { (0..n).map(|i| f(i as f64)).collect() };
    let a = col(&|i| (0.9 * i + 0.2).sin());
    let xi = col(&|i| (2.1 * i + 1.3).sin());
    let w = col(&|i| (3.7 * i + 0.5).sin());
    let y: Vec<f64> = (0..n)
        .map(|i| a[i] + (5.3 * i as f64 + 0.9).sin())
        .collect();
    let near = Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("a".into(), Column::Numeric(a)),
            (
                "xi2".into(),
                Column::Numeric(xi.iter().zip(&w).map(|(x, w)| x + 1e-7 * w).collect()),
            ),
            ("xi".into(), Column::Numeric(xi)),
            (
                "g".into(),
                Column::factor_from_labels(
                    &(0..n)
                        .map(|i| format!("g{:02}", i / 10))
                        .collect::<Vec<_>>(),
                ),
            ),
        ],
        n,
    };
    for f in ["y ~ a + xi + xi2", "y ~ a + xi + xi2 + (1 | g)"] {
        check(
            f.to_string(),
            formula::fit(f, &near, Family::Gaussian, &opts),
        );
    }
    assert_eq!(gaussian.len() + 2, 22);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
