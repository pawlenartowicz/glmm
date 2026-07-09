# Changelog

All notable changes to the `glmm` crate are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this crate predates its
first release (no published version yet), so entries accumulate under *Unreleased*
until the first tag.

## [Unreleased]

### Prior weights (sparse binomial GLMM)

#### Added

- **`FitOptions::weights`** — per-row prior (case) weights, lme4's `weights=`.
  Currently honored only on the sparse binomial GLMM path; every other dispatch
  rejects them at the fit boundary rather than silently fit unweighted. An
  aggregated binomial (y = success proportion, weight = trial count) now fits
  directly — lme4's `cbind(s, m−s)` objective, which shares its argmin (and so
  β/SE/varcomp) with the expanded-Bernoulli fit — letting the
  `sim_sparse_binomial` parity rung fit its 240 aggregated rows instead of the
  3,059-row Bernoulli expansion. Parity holds at unchanged tolerances; the
  per-solve O(n·width²) cost collapses accordingly.

### Two-stage GLMM optimizer

#### Changed

- **GLMM fits now use a two-stage optimizer** (lme4's structure, Bates et al. 2015
  §3): a fast θ-only search profiles the fixed effects β out per PIRLS iteration,
  then a short joint (θ, β) polish on the exact Laplace objective warm-started from
  it. The converged (θ̂, β̂) and all standard errors are unchanged up to optimizer
  tolerance — the parity goldens hold at their existing tolerances — but the outer
  evaluation count drops materially (roughly 2× fewer BOBYQA evaluations on the
  grouseticks 3-crossed Poisson fixture). The prior single-stage joint solve remains
  available as an internal A/B toggle. `Fit::n_eval` now includes stage-1
  evaluations, so eval counts are not directly comparable to versions before this
  change.

#### Added

- **PIRLS step-halving.** The inner penalized-IRLS loop now backtracks (halves the
  step, up to 10 times) when a full Fisher-scoring step raises the penalized
  deviance, hardening convergence on ill-scaled joint (u, β) steps; an exhausted
  backtrack surfaces as the existing non-converged/NaN failure state.

### M3.5 — warm-start entry-split

The fit surface now separates model *structure*, optimizer *warm-start state*, and
method *knobs* into three distinct places (`docs/GLMM/api.md`, Layers A–C). The
stable `fit`/`fit_grouped` signatures are unchanged; the breakage is in the shapes
they consume.

#### Changed (breaking)

- **`ModelSpec` is structure-only.** Removed the method knobs `wald_se` and `nagq`
  and every magnitude payload — a `ModelSpec` can no longer carry a start estimate.
  - `ReStructure` and `Grouping` lost `tau_squared` and now hold
    `slopes: Vec<ColumnId>` (the `SlopeTerm` struct, which bundled a column with its
    variance/correlation magnitudes, is deleted along with the `re_correlation_*`
    helpers).
  - `Family::Gamma` lost its `dispersion: Option<f64>` payload;
    `Family::NegativeBinomial` lost its `theta: Option<f64>` payload.
- **`FitOptions` gained the relocated knobs:** `wald_se`, `nagq`, and `dispersion`
  (the Gamma fix-vs-estimate directive). All are defaulted — construct with
  `..FitOptions::default()` (Wald SE `Hessian`, `nagq` 1 = Laplace, `dispersion`
  `None` = estimate φ post-fit). `FitOptions` now implements `Default`.
- **The stable `fit`/`fit_grouped` cold-start the optimizer** — they no longer derive
  a warm start from spec magnitudes; the kernels use their `THETA0` blind start. The
  converged MLE is unchanged up to optimizer tolerance (start-independent), so the
  oracle goldens stay green at their existing tolerances.
- **Cargo feature `mcpower` renamed to `loop_advanced`.** Capability-named rather
  than consumer-named; still off by default, still the unstable scratch-explicit
  loop-tier surface with no semver guarantees. The `cluster_theta_truth` re-export is
  removed (truth-start magnitudes no longer live in `ModelSpec`).

#### Added

- **`StartValues { beta, theta }`** — the warm-start primitive (api.md Layer B): raw
  optimizer state (`beta` = fixed-effect start, `theta` = RE Cholesky parameters),
  not high-level variances. Exported `pub` only behind the `loop_advanced` feature;
  the stable tier never takes it. Carries no `phi`/`nb_theta`: Gamma φ is profiled and
  the GLMM neg-binomial θ search is a global bracket, so neither warm-starts anything
  reachable through the loop surface.

#### Migration — MCPower pin-bump action

MCPower consumes a pinned published `glmm`, so this rename is not a live break. When
MCPower next bumps its pinned `glmm`:

- switch its feature selection `mcpower` → `loop_advanced`;
- build any spec-derived start as a raw `StartValues.theta` (column-major vech of the
  RE Cholesky parameters) instead of relying on the removed `cluster_theta_truth` /
  `ModelSpec` magnitude fields.
