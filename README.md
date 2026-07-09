# glmm

[![CI](https://github.com/pawlenartowicz/glmm/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/glmm/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/glmm.svg)](https://crates.io/crates/glmm)
[![docs.rs](https://img.shields.io/docsrs/glmm)](https://docs.rs/glmm)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)

**Standalone f64 GLM(M) fit kernels — OLS → GLM → LMM → GLMM — in pure Rust on faer.**

Fits fixed-effect and mixed (random-intercept/random-slope) models for
Gaussian, Binomial (logit/probit), Poisson, Gamma, and Negative-Binomial
outcomes, validated against R/lme4 and Julia/MixedModels.jl goldens.

**New to the crate? Start with [`TUTORIAL-RUST.md`](TUTORIAL-RUST.md)** — a
single-page, three-layer walkthrough (cold fit → warm fit → advanced loop)
plus a short section on parsing an R-style formula string instead of building
inputs by hand. (A Python port is planned; its tutorial will live alongside
this one as `TUTORIAL-PYTHON.md`.)

## Alpha (0.0.x)

The stable surface is [`fit_cold`]/[`fit_warm`] + `ModelSpec` + `GroupIds`.

| Model                      | Fixed-only (`re: None`) | Mixed (`re: Some`)                                          |
|-----------------------------|--------------------------|---------------------------------------------------------------|
| Gaussian                    | OLS                      | LMM — dense, or sparse-Z when an extra grouping carries a random slope |
| Binomial (logit/probit)     | GLM                      | GLMM — dense only                                              |
| Poisson (log)                | GLM                      | GLMM — dense only                                              |
| Gamma (log/inverse)          | GLM                      | GLMM — dense only                                              |
| Negative-Binomial (log)      | GLM                      | GLMM — dense only                                              |

A mixed non-Gaussian model that falls outside the dense solver's envelope
(too many extra groupings, or an extra grouping too wide) is not yet
implemented and panics rather than silently misrouting — see
[`TUTORIAL-RUST.md`](TUTORIAL-RUST.md) for the exact envelope.

The `loop_advanced` cargo feature (off by default) exposes an unstable
scratch-explicit hot-path surface for warm-start callers like MCPower's
simulation loop — **no semver guarantees; do not depend on it outside a
pinned revision.**

## Quick example

```rust
use glmm::{fit_cold, Family, FitOptions, GroupIds, ModelSpec, ReStructure, Sizing};

// y ~ x + (1 | group), Gaussian — a random-intercept LMM.
let model = ModelSpec {
    family: Family::Gaussian,
    re: Some(ReStructure {
        sizing: Sizing::FixedClusters { n_clusters: 6 },
        slopes: vec![],
        extra_groupings: vec![],
    }),
};
let ids = GroupIds { primary: vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], extra: vec![] };
let opts = FitOptions { target_indices: vec![0, 1], ..Default::default() };

let fit = fit_cold(&x, &y, n, p, &model, &ids, &opts);
assert!(fit.converged);
```

See [`TUTORIAL-RUST.md`](TUTORIAL-RUST.md) for the full walkthrough: warm
starts, the advanced hot-loop surface, and building `x`/`ModelSpec`/`GroupIds`
from a formula string instead of by hand.

## Design

| Property       | Detail                                                              |
|----------------|--------------------------------------------------------------------|
| No `unsafe`    | zero `unsafe` (workspace baseline `unsafe_code = "warn"`)           |
| Deterministic  | no RNG, no global state, no I/O in the fit path                     |
| Linear algebra | [`faer`](https://crates.io/crates/faer) 0.24, pinned               |
| MSRV           | Rust 1.85 (floor set by the `bobyqa` dep)                          |

## Origin

`glmm` was carved out of [MCPower](https://github.com/pawlenartowicz/)'s
simulation engine — the numerics here are the same parity-pinned kernels that
power MCPower's Monte Carlo fits, split out so they're usable standalone. The
`loop_advanced` feature above exists specifically to serve MCPower's
warm-start hot loop as a consumer of this crate.

## License

`GPL-3.0-or-later` (coupled to the GPL-3 MCPower flagship).

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
