# Installing glmm

## Rust

```bash
cargo add glmm
```

MSRV is Rust 1.85 — the floor is set by the `bobyqa` dependency. Linear
algebra runs on [`faer`](https://crates.io/crates/faer) 0.24, pinned.

The semver-covered surface is `fit_cold`/`fit_warm` + `ModelSpec` +
`GroupIds`.

Default features build the `formula` frontend (R-style formula strings, the
only thing that pulls in `regex`) — everything else is off by default. The
`loop_advanced` feature exposes an unstable, scratch-explicit hot-path
surface for warm-start callers like MCPower's simulation loop, with no
semver guarantees. The `orchestrate` feature exposes the string-typed fit
orchestration the Python and R packages are built on (the family/link string
vocabulary, the formula-and-data lowering, and the flattened result both
ports publish), also with no semver guarantees: its shape follows the ports'
needs. The `parallel` feature (experimental) enables in-fit
parallelism via rayon — the AGQ cluster loop and the FD-Hessian SE grid —
gated at runtime by `FitOptions::parallel_inner`; both the feature and the
flag must be on for any thread to spawn, and it's a no-op on wasm32.

## Python

```bash
pip install glmm
```

Requires Python 3.10+. The only runtime dependency is NumPy — no
BLAS/LAPACK system dependency.

## R

From r-universe:

```r
install.packages("fastglmm", repos = c("https://pawlenartowicz.r-universe.dev", getOption("repos")))
```

From a checkout — needs Rust (`cargo` and `rustc >= 1.85` on the `PATH`):

```r
# in GLMM/r/
install.packages(".", repos = NULL, type = "source")
```

For building a distributable, self-contained source tarball (e.g. for CRAN),
see the Development section of [r/README.md](../r/README.md), which covers
`tools/cran-tarball.sh`.
