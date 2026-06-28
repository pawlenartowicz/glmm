# glmm

**Standalone f64 GLMM fit kernels — OLS → GLM → LMM → GLMM — in pure Rust on faer.**
The parity-pinned numerics from the [MCPower](https://github.com/pawlenartowicz/) engine, usable on their own.

## Alpha (0.0.x)

The stable surface is `fit(...)` + `ModelSpec`. Today `fit` wires **OLS** and **LMM**; GLM and GLMM
exist as kernels but are **not yet exposed** through `fit` (calling them panics `unimplemented!`) and land
in a later release. The `mcpower` cargo feature (off by default) exposes an unstable scratch-explicit
hot-path API for simulation callers — **no semver guarantees; do not depend on it.**

## Design

| Property       | Detail                                                              |
|----------------|--------------------------------------------------------------------|
| No `unsafe`    | zero `unsafe` (workspace baseline `unsafe_code = "warn"`)           |
| Deterministic  | no RNG, no global state, no I/O in the fit path                     |
| Linear algebra | [`faer`](https://crates.io/crates/faer) 0.24, pinned               |
| MSRV           | Rust 1.85 (floor set by the `bobyqa` dep)                          |

## License

`GPL-3.0-or-later` (coupled to the GPL-3 MCPower flagship).

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
