# glmm — algorithmic design rationale

*The main algorithmic differences between `glmm` and the reference engines
(lme4, MixedModels.jl), and why each one makes it faster.
Written for readers who already know how mixed models are fit. Code-level
detail: [`algorithms.md`](algorithms.md),
[`algorithms-lmm.md`](algorithms-lmm.md),
[`algorithms-glmm.md`](algorithms-glmm.md).*

All engines minimise the same objective — the profiled deviance over the
relative-Cholesky θ, derivative-free. `glmm` changes how each evaluation is
computed and what happens at the edges, not what is estimated: point estimates
agree with lme4/MixedModels.jl to the validation gates (~1e-3 relative on β and
variance components) on every supported design.

## The differences

1. [Sufficient statistics instead of a Z matrix](#1-sufficient-statistics-instead-of-a-z-matrix)
2. [Per-shape kernels behind one dispatch](#2-per-shape-kernels-behind-one-dispatch)
3. [Zero-allocation workspace and warm starts](#3-zero-allocation-workspace-and-warm-starts)
4. [Measured optimizer schedule](#4-measured-optimizer-schedule)
5. [Fused SIMD transcendentals](#5-fused-simd-transcendentals)
6. [Standard errors from a tightly re-converged Hessian](#6-standard-errors-from-a-tightly-re-converged-hessian)
7. [Deterministic edge-case policy](#7-deterministic-edge-case-policy)
8. [A pure kernel](#8-a-pure-kernel)

## 1. Sufficient statistics instead of a Z matrix

**What.** On the dense path, `glmm` never materialises the random-effects
design Z. One pass over the data accumulates per-cluster Grams; every θ
evaluation afterwards works on those cluster-level blocks. lme4 and
MixedModels.jl build a sparse Z and factor `ΛᵀZᵀZΛ + I` against data-sized
structures at every evaluation.

**Why it is faster.** Fit cost is dominated by objective evaluations (BOBYQA
needs tens to hundreds). Making each evaluation scale with the number of
clusters instead of the number of rows removes N from the inner loop entirely.
For balanced single-intercept designs the evaluation collapses further, to
closed-form arithmetic. The optimum is unchanged because the objective is the
same — only the factorisation route differs.

## 2. Per-shape kernels behind one dispatch

**What.** One entry point routes each design to the cheapest kernel that fits
it: closed-form collapse (balanced single intercept), block-diagonal PIRLS (no
extra groupings), a core-plus-Schur factorisation (intercept-only
crossed/nested extras), and a sparse-Z solver for everything over the dense
envelope. lme4 and MixedModels.jl run one general solver for all shapes.

**Why it is faster.** The common simulation shapes get a
kernel specialised to their structure instead of paying general-solver
overhead. The router's contract also means a design outside the
dense envelope is *redirected* to the sparse solver, never rejected — there is
no reachable `unimplemented!` in the dispatch. The dense/sparse split cannot
introduce disagreement: at the same θ both kernels factor the same SPD system,
so by Cholesky uniqueness they produce the same fit to machine precision.

## 3. Zero-allocation workspace and warm starts

**What.** All solver scratch lives in one workspace sized by model *shape*
(groupings, family, p, max rows), not by data values. Refitting the same shape
on new data allocates nothing; caller-supplied β/θ start values feed the
optimizer directly. The `loop_advanced` feature exposes this reuse explicitly.

**Why it is faster.** This targets the regime `glmm` was built for: thousands
of refits of one shape (power simulation, resampling). Allocation, setup, and
cold-start cost are paid once per shape instead of once per fit. R-based
engines rebuild model structures per call; autodiff engines (TMB/glmmTMB) pay
per-fit tape/setup that this regime never amortises. The claim is scoped to
the loop — no single-fit benchmark claim is made.

## 4. Measured optimizer schedule

**What.** The BOBYQA configuration is tuned, not default: interpolation-set
size `⌈1.5n⌉+1` instead of Powell's `2n+1` (from `n_θ ≥ 3`), a trust-radius
schedule scaled to the start point, a stopping radius relaxed from `1e-8` to
`1e-6`, and for GLMMs a two-stage search (θ-only with β profiled inside PIRLS,
then a joint `[θ|β]` polish that alone decides convergence).

**Why it is faster.** Every choice was swept against the validation corpus and
kept only where it cut evaluations without moving any gated result: the `npt`
mid-size won on every dimension ≥ 3, the relaxed stopping radius saved ~25% of
evaluations at measured-equivalent accuracy, and stage 1 is a pure warm-start
accelerant (skipping it is bit-identical, just slower). The reported optimum
is always the full Laplace one.

## 5. Fused SIMD transcendentals

**What.** The binomial-logit inner loop computes probability, working weight,
and the deviance fold in one vectorised pass over crate-own minimax `exp`/
`log1p` kernels (≤ 2 ULP against libm), sharing one `exp(−|η|)` per row.

**Why it is faster.** IRLS/PIRLS time is dominated by link-function
transcendentals. Fusing them removes redundant `exp` calls and keeps the loop
in SIMD registers; the ≤ 2 ULP bound keeps the result within the validation gates.

## 6. Standard errors from a tightly re-converged Hessian

**What.** The default GLMM covariance is the finite-difference Hessian of the
Laplace deviance (lme4's `use.hessian = TRUE` method), with PIRLS re-converged
at a tolerance two decades tighter than the fit tolerance at every perturbed
point. The conditional-on-θ̂ alternative (`Rx`, MixedModels.jl's only method)
is also available.

**Why it is more accurate.** An FD Hessian is only as good as the objective
under it: `glmer` at its default `tolPwrss` evaluates the Hessian on a surface
whose working weights lag the mode by one iteration, which biases its own
Hessian SEs by ~1%. `glmm`'s tight re-convergence makes the second differences
step-invariant by construction; against an artifact-free lme4 oracle
(tightened `tolPwrss`) the SEs agree to ≤ 2e-5 relative. Offering both arms
also means either reference engine can be matched exactly.

## 7. Deterministic edge-case policy

**What.** Every failure mode has a defined, deterministic output instead of a
warning or an exception:

- Rank-deficient X: aliased columns detected in R's `dqrdc2` order, dropped,
  the reduced model fit, `NaN` returned exactly where lme4 puts `NA`.
- Ill-conditioned but not redundant X: fitted as-is — real β̂/SE, no column
  dropped — with an `IllConditioned` note on `Diagnostics::notes` naming the
  worst-conditioned column. Large standard errors carry the honest signal that
  the columns are barely separable; this is distinct from the rank-deficient
  case above, which is exact aliasing.
- Variance component at zero: pinned to exactly `0.0` (every diagonal
  component ≤ 1e-4), fit still counted converged, reported through
  `Diagnostics::singular`/`Diagnostics::pinned`.
- Optimizer cap-out on a flat plateau: the best finite point is returned with
  `converged: false`, not a NaN fill.
- Genuine numerical failure: NaN-filled, non-converged — never a crash.

**Why it matters.** A simulation loop cannot stop to read console
warnings. Machine-readable flags with exact-zero pins (floating-point stable
across platforms) let a caller classify every fit programmatically, and the
boundary between "converged at an edge" and "failed" is explicit instead of
inferred from warning text.

## 8. A pure kernel

**What.** The fit path has no RNG, no global state, no I/O, and no `unsafe`.
The optional in-fit parallelism is bit-identical to serial by design (each
task writes one pre-assigned slot; reductions run in fixed order). The kernel
compiles unmodified to `wasm32` (CI-gated), and the Python and R packages are
thin wrappers over the same compiled code.

**Why it matters.** Same inputs, same bits — on any platform, any
thread count, any wrapper. That is what makes results reproducible across a
simulation grid, lets the parallel feature share the serial test goldens, and
puts the engine in browsers and embedded runtimes that R and Julia cannot
reach.

## What keeps the speed honest

Every difference above is held to the frozen validation oracle: identical models
fit in R/lme4 and Julia/MixedModels.jl — two independent implementations — on
27 committed datasets from single-intercept LMMs to crossed/nested/sparse
GLMMs. References are never regenerated to relax a tolerance; a disagreement
beyond the agreement band is investigated as a `glmm` bug first and passes only
once it is written up and registered (`validation/divergences.json`). Shapes only lme4 covers (Gamma/NB/probit
GLMMs, AGQ) are pinned by committed lme4 goldens.

## Limits

The trade is inference machinery and breadth: no BLUPs/`predict`, no
profile/bootstrap CIs, no small-sample df corrections; fewer families than
glmmTMB/GLMMadaptive; REML-only LMMs; and past a few dozen θ parameters,
gradient-based engines are expected to win. For one interactive model on one
dataset, lme4 or its neighbours remain the right tool.
