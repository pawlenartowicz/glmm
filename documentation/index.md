# glmm documentation

`glmm` is a standalone f64 mixed-model fit kernel — OLS → GLM → LMM → GLMM, in
pure Rust on [`faer`](https://crates.io/crates/faer) — fitting fixed-effect and
mixed (random-intercept/random-slope) models for Gaussian, Binomial
(logit/probit/cloglog), Poisson, Gamma, Negative-Binomial, and (fixed-effect
GLM only) Inverse-Gaussian outcomes, validated
against R/lme4 and Julia/MixedModels.jl goldens. It is derivative-free (BOBYQA
over the profiled deviance, no autodiff), REML-locked for Gaussian mixed
models (no ML switch), validation-pinned against a frozen lme4/MixedModels.jl
oracle that is never relaxed to match a disagreement, and WASM-ready (the fit
path carries no `unsafe`, no RNG, no global state, no I/O, and compiles
unmodified to `wasm32`). It deliberately is **not** a general inference
toolkit: no BLUPs/`predict`, no profile or bootstrap confidence intervals, no
small-sample degrees-of-freedom corrections, and fewer families than
glmmTMB/GLMMadaptive — anything the engine cannot compute honestly errors with
the reason instead of guessing, and for one interactive model on one dataset,
lme4 or its neighbours remain the right tool. What it is built for instead is
thousands of refits of one model shape — power simulation, resampling — through
a zero-allocation, warm-startable workspace.

## Pick your language

- **Rust** — [`tutorial-rust.md`](tutorial-rust.md): cold fit → warm fit →
  advanced hot loop, plus the formula frontend.
- **Python** — [`tutorial-python.md`](tutorial-python.md): the `glmm` package
  walkthrough.
- **R** — [`tutorial-r.md`](tutorial-r.md): the `fastglmm` package walkthrough.

## Full doc map

| Tier | File | Purpose |
|---|---|---|
| Entry | [`installation.md`](installation.md) | Installing the Rust crate, Python package, and R package |
| Tutorials | [`tutorial-rust.md`](tutorial-rust.md) | Three-layer Rust walkthrough: cold fit → warm fit → advanced loop, plus the formula frontend |
| Tutorials | [`tutorial-python.md`](tutorial-python.md) | The Python package (`glmm`) walkthrough |
| Tutorials | [`tutorial-r.md`](tutorial-r.md) | The R package (`fastglmm`) walkthrough |
| Examples | [`examples-python.md`](examples-python.md) | Worked recipes against real, frozen datasets, output checked against pinned lme4 results |
| Examples | [`examples-r.md`](examples-r.md) | The same recipes as `examples-python.md`, in R |
| Migration | [`coming-from-lme4.md`](coming-from-lme4.md) | Call mapping from lme4, what's deliberately missing, and behavioral differences to watch (covers both the R and Python surface) |
| Migration | [`coming-from-statsmodels.md`](coming-from-statsmodels.md) | Migrating from `statsmodels` `MixedLM`/`GLM` to `glmm.fit` (Python only) |
| Reference | [`supported_families.md`](supported_families.md) | Family × link support matrix, canonical-link notes, dispersion conventions |
| Reference | [`formula.md`](formula.md) | What the formula parser accepts and rejects, with workarounds |
| Reference | [`conventions.md`](conventions.md) | Estimation, standard-error, dispersion, and variance-component conventions, and the flags on a fit result |
| Reference | [`troubleshooting.md`](troubleshooting.md) | Fixes for singular fits, non-convergence, `NotImplementedError`, and rejected formulas |
| Reference | [`validation.md`](validation.md) | How glmm is validated against lme4 and MixedModels.jl, what's covered, and known tolerances/exemptions |
| Internals | [`algorithms.md`](algorithms.md) | Algorithm map entry point: full dispatch graph, knob index, OLS/GLM paths |
| Internals | [`algorithms-lmm.md`](algorithms-lmm.md) | LMM: θ-Cholesky, profiled REML, closed-form shortcut, BOBYQA, boundary handling |
| Internals | [`algorithms-glmm.md`](algorithms-glmm.md) | GLMM: PIRLS, Laplace vs AGQ, dense vs sparse Z, NB outer loop, warm starts |
| Internals | [`glmm-design.md`](glmm-design.md) | Algorithmic design rationale: the differences from lme4/MixedModels.jl and why each one is faster |
