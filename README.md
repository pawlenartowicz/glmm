# glmm

[![CI](https://github.com/pawlenartowicz/glmm/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/glmm/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/glmm.svg)](https://crates.io/crates/glmm)
[![PyPI](https://img.shields.io/pypi/v/glmm.svg)](https://pypi.org/project/glmm/)
[![r-universe](https://pawlenartowicz.r-universe.dev/badges/fastglmm)](https://pawlenartowicz.r-universe.dev/fastglmm)
[![docs.rs](https://img.shields.io/docsrs/glmm)](https://docs.rs/glmm)
[![License: LGPL-3.0-or-later](https://img.shields.io/badge/license-LGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)

**`glmm`: mixed models in Rust, Python and R, from one engine.**

**glmm** fits ordinary least squares, generalized linear models, linear mixed
models and generalized linear mixed models. One engine, written in Rust, sits
behind the Rust crate, the Python package (`glmm`) and the R package
(`fastglmm`), and also runs in the browser through WASM.

## The idea

- **Accuracy first.** glmm reaches the same log-likelihood as lme4, glmmTMB,
  MixedModels.jl and GLMMadaptive, or a better one.
- **Multi-solver.** Each model gets a method that suits it. Least squares uses
  a closed form, and GLMs use IRLS. Mixed models use BFGS where glmm has a cheap
  exact gradient, and BOBYQA, which needs no gradient, elsewhere.
- **Fast.** glmm is faster than these four packages on every model in the speed
  tests, and it uses little memory.

## Pick your language

**Rust**: [Rust tutorial](documentation/tutorial-rust.md)

```bash
cargo add glmm
```

**Python** (`glmm`): [Python README](python/README.md)

```bash
pip install glmm
```

**R** (`fastglmm`): [R README](r/README.md)

```r
install.packages("fastglmm", repos = c("https://pawlenartowicz.r-universe.dev", getOption("repos")))
```

## Validation

glmm is checked on 923 test models before every release. They cover the
Gaussian, binomial, Poisson, Gamma and negative binomial families, random
intercepts, correlated slopes, nested and crossed groupings, and data from 60 to
30,000 rows. Each model is also fitted by lme4, glmmTMB, MixedModels.jl and
GLMMadaptive, where they support it. glmm is compared with the best of their
results, not with one chosen package. If glmm reaches a worse log-likelihood
than that best result, the release fails. If its estimates differ from the
other packages, the difference passes only once it is explained in writing.

Standard errors of mixed models come from the exact second derivatives of the
likelihood, not from a finite-difference approximation. On the test models they
differ from a high-precision lme4 run by at most 0.002% (relative).

glmm was also stress-tested on many hard and borderline models. On such models
it returns a result with flags that say what happened, instead of only printing
a warning. A variance that falls to zero is set to exactly zero and flagged. A
redundant column is dropped and marked, as in lme4. When a fit fails, the
`converged` flag says so. glmm does not crash.

The package versions, the checks and how to rerun them are in
[`validation.md`](documentation/validation.md).

## Multi-solver paradigm

glmm combines several optimizers for mixed models. Where it has a cheap exact
gradient, it uses BFGS. Elsewhere it uses BOBYQA, which needs no gradient.
BOBYQA also refines the BFGS result and takes over when BFGS fails.

On larger GLMMs, a fast approximate pass first finds a good starting point, and
the full search starts from there. When the main search does not converge, the
fit tries a second search strategy from the same start and keeps the better
answer.

The linear algebra also uses more than one method. Common model shapes use
small dense blocks, one per group. Larger or more complex designs go to a
sparse solver instead of being refused. Given the same variance parameters,
both return the same answer. The choice affects speed only.

## Speed

glmm moves as little data as it can. One pass over the data builds a small
summary per group, and every later step of the search works on those summaries.
The cost of each step therefore grows with the number of groups, not the number
of rows.

glmm is also designed for refitting the same model many times. Its working
memory depends only on the shape of the model. So refitting the same model on
new data, as in a power simulation or a bootstrap, needs no new memory. Each
refit can also start from the previous answer.

The search also needs few steps. Its settings were tuned on the 923 test models
and kept only where they cut the number of steps without changing any result.
With a cheap exact gradient it needs even fewer.

Each of these choices, and how it differs from lme4 and MixedModels.jl, is in
[`glmm-design.md`](documentation/glmm-design.md).

## What it fits

| Family | Links | Mixed models |
|---|---|---|
| Gaussian | identity | yes |
| binomial | logit, probit, cloglog | yes |
| Poisson | log | yes |
| Gamma | log, inverse | yes |
| negative binomial | log | yes |
| inverse Gaussian | log, inverse squared | no, GLM only |

- **Random effects:** random intercepts, correlated random slopes, and several
  grouping factors, nested or crossed.
- **Linear mixed models:** REML (the default) or maximum likelihood.
- **Generalized linear mixed models:** the Laplace approximation (the default)
  or adaptive quadrature. Adaptive quadrature works on binomial, Poisson,
  negative binomial and Gamma models with one grouping factor and up to 3
  random effects per group.
- **Prior weights and offsets.**
- **Tests:** tests of the fixed terms, and comparison of two nested models
  (`anova` and `compare` in the Python and R packages), described in
  [`hypothesis-tests.md`](documentation/hypothesis-tests.md).

Details per family are in
[`supported_families.md`](documentation/supported_families.md).

## Cargo features

| Feature | Default | What it does |
|---|---|---|
| `formula` | on | R-style formula strings. The only feature that pulls in `regex`. |
| `parallel` | off | Experimental. Runs parts of a fit on several threads (rayon). Results are the same as on one thread. |
| `loop_advanced` | off | Unstable. A low-level API for refitting one model many times, as in a simulation loop. |

For the kernel only, with no formula parser and no `regex`, turn the defaults
off: `glmm = { version = "0.6", default-features = false }`. Details are in
[`installation.md`](documentation/installation.md).

## WASM and stability

The engine compiles to `wasm32` unchanged, and CI checks that it does. A Pyodide
wheel of the Python package is attached to each
[GitHub release](https://github.com/pawlenartowicz/glmm/releases). The same
model fitted on Linux, on macOS or in WebAssembly gives the same fit: across the
test models the log-likelihoods differ by at most 3e-7.

glmm is in beta. Semver covers `fit_cold`, `fit_warm`, `ModelSpec` and
`GroupIds`. The `formula` frontend stays outside it until 1.0.0, and
`loop_advanced` has no semver guarantee. The minimum supported Rust version is
1.85.

## Documentation

All pages are listed in [`documentation/index.md`](documentation/index.md). The
main ones:

- Tutorials: [Rust](documentation/tutorial-rust.md),
  [Python](documentation/tutorial-python.md), [R](documentation/tutorial-r.md)
- Coming from another package: [lme4](documentation/coming-from-lme4.md),
  [statsmodels](documentation/coming-from-statsmodels.md)
- Reference: [formulas](documentation/formula.md),
  [conventions](documentation/conventions.md),
  [warnings](documentation/warnings.md),
  [troubleshooting](documentation/troubleshooting.md)
- How it works: [algorithms](documentation/algorithms.md),
  [design](documentation/glmm-design.md),
  [validation](documentation/validation.md)

## Citation

If you use glmm in research, please cite:

Lenartowicz, P. (2026). glmm: Mixed-model fitting in Rust, Python and R [Computer software]. https://github.com/pawlenartowicz/glmm

```bibtex
@software{glmm2026,
  author = {Lenartowicz, Pawe{\l}},
  title  = {{glmm}: Mixed-Model Fitting in Rust, Python and R},
  year   = {2026},
  url    = {https://github.com/pawlenartowicz/glmm}
}
```

## License

`LGPL-3.0-or-later`. See [`LICENSE`](LICENSE) and [`LICENSE-GPL`](LICENSE-GPL).

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
