# How glmm is validated

## Two independent reference engines

Every validated model is fit three ways: on the same data, with the same
formula, in R `lme4` and in Julia `MixedModels.jl`, then compared against
`glmm`. Two independently-implemented engines agreeing with each other
within tolerance is the truth condition `glmm` is held to — not agreement
with either engine alone.

## The oracle is sacred

The reference data and reference results are frozen and treated as ground
truth. When `glmm` disagrees with them, `glmm` is presumed wrong: the fix
goes into `glmm`, not the reference. A reference result is only regenerated
when the reference's own model spec is proven wrong (wrong formula, family,
or link), and that requires a recorded justification. Tolerances are never
relaxed to make `glmm` pass. Where the two reference engines disagree with
each other beyond tolerance, that is recorded as a flag to investigate — the
harness never silently picks whichever one is closer to `glmm`.

## What is covered

The suite spans Gaussian, binomial, Poisson, and Gamma models across a
range of random-effect shapes (intercept-only, correlated slopes, nested
and crossed grouping, canonical and non-canonical links, dense and sparse
routing). The exact dataset list and per-dataset model
spec is the single source of truth in
[`../validation/manifest.json`](../validation/manifest.json); prior (case)
weights get their own manifest rungs (`tier: "weights"`, rungs 29-43).
The Python and R packages are not compared against lme4/MixedModels.jl
directly — they wrap the same Rust kernel, so they are gated against the
Rust engine's own results at a round-off tolerance, confirming the wrapper
introduces no numerical drift.

## Tolerances and known exemptions

Tolerances are per-quantity, not a single global threshold, because point
estimates and standard errors have different natural agreement bands: fixed
effects and variance-component standard deviations are compared at a
relative tolerance, the log-likelihood at an absolute tolerance on the
shared scale (looser for GLMM than LMM, since GLMM compares two different
Laplace-approximation optimizers on the same objective rather than a
near-exact profiled criterion), and standard errors similarly at a relative
tolerance. All bands were set from the measured worst case at freeze plus a
margin, and are never widened to accommodate a failure.

The one recorded, permanent exemption is the GLMM standard-error method
split: `glmm` and lme4 both compute a Hessian-based standard error (which
keeps the θ–β coupling), but MixedModels.jl computes only the Rx variant
(conditional on the estimated variance components). The comparison matches
method to method — Rx against Rx, Hessian against Hessian — rather than
comparing across methods, which would manufacture a spurious disagreement.
Where lme4 and MixedModels.jl disagree with each other on a shared
quantity, that is recorded as a flag for investigation, never resolved by
picking whichever reference happens to sit closer to `glmm`.

## Running it yourself

From `validation/`, `./run.sh` fits `glmm` (Rust) and its Python and R ports
and compares them against the existing reference results on disk — R and
Julia are not refit, so this is the fast path for iterating on `glmm`
itself. `./run.sh --oracles` refits all engines, including the R and Julia
references, for when the oracle itself needs regenerating. See
[`../validation/README.md`](../validation/README.md) for the full directory layout,
result schema, and running instructions.
