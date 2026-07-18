# Changelog

All notable changes to the `glmm` crate are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The Python package (`glmm` on PyPI) is versioned in lockstep with the crate and
shares these entries; Python-specific notes are called out where they differ.

## [Unreleased]

## [0.1.1] — 2026-07-18

Additive release: an offset term and post-fit reporting fields (log-likelihood,
AIC/BIC df, fitted means, conditional modes). Nothing on the stable surface
changed shape, so every 0.1.0 fit keeps its result up to optimizer tolerance;
the oracle goldens hold at their existing tolerances.

### Added

- **`FitOptions::offset`** — a per-row additive term on the linear-predictor
  scale, `η = offset + Xβ (+ Zb)`, matching R's `glm(offset=)` / `glmer(offset=)`.
  A fixed known contribution, not a parameter (β must not absorb it); the
  canonical use is a Poisson exposure, `offset = log(exposure)`. Supported on
  every path — OLS, GLM, LMM, GLMM (dense and sparse) — with identity-link
  paths applying it as an exact `y − o` shift and `Fit::fitted` still reporting
  means on the original `y` scale. Also on the Python `fit(offset=)`. A new
  `sim_poisson_offset` parity rung (28) pins it against `glmer(offset=)`. The R
  port (`fastglmm`) still rejects `offset=` / `offset()` by design.
- **`Fit::loglik`, `Fit::df`, `Fit::reml`** — the log-likelihood at the fitted
  parameters, `deviance` with its dropped data-only constants restored onto
  lme4's `logLik()` scale, the AIC/BIC parameter count, and a flag marking the
  LMM REML criterion. Together they give `AIC = 2·df − 2·loglik` and
  `BIC = df·ln(n) − 2·loglik` on every path. `loglik` matches `lme4::logLik`
  including the aggregated-binomial `cbind(s, m−s)` form under `weights=`; on
  the Gaussian LMM paths it is the REML criterion `−REMLcrit/2`, comparable only
  between models with identical fixed effects — `reml` is set there, mirroring
  lme4's REML-fit `anova` warning. `df` counts retained fixed effects (lme4's
  NA-coefficient handling for aliased columns) + RE θ parameters + 1 where the
  family estimates a dispersion/scale.
- **`Fit::fitted`** — fitted means `μ̂` per row through the inverse link (lme4
  `fitted()`). Empty on non-converged fits and on the Gaussian LMM paths, which
  fit via sufficient statistics and never materialize per-row means.
- **`Fit::ranef`, `Fit::ranef_levels`** — random-effect conditional modes `b̂`
  (BLUPs), one block per grouping in `varcorr`/`re_groups` order, level-major,
  with `ranef_levels` giving each grouping's level count for slicing. Empty on
  the same paths as `fitted`.
- All six new fields cross the Python and R shims onto their fit results
  (`Fit.loglik`/`df`/`reml`/`fitted`/`ranef`/`ranef_levels` in Python).
- Seven user-facing guides under `documentation/`: `installation.md`,
  `formula.md`, `conventions.md`, `coming-from-lme4.md`, `glmm-design.md`,
  `validation.md`, `troubleshooting.md`. The Python and R READMEs now link them
  instead of inlining the formula and factor-coding rules.

### Changed

- **The boundary (singular) fit warning now names the degenerate components.**
  lme4's exact text (`boundary (singular) fit: see help('isSingular')`) is
  extended with `sd(term | group) = 0` per collapsed variance and
  `corr(a, b | group) = ±1` per degenerate correlation. Exact comparisons are
  safe because the kernel pins boundary components to exact 0 / ±1; the bare
  lme4 text is kept when only the relative-tolerance singular check fired. The
  Python and R ports emit the same extended message.

## [0.1.0] — 2026-07-16

First release of the Python package (`glmm` on PyPI), and the first crate
release since `0.0.2`. The breaking changes below are real breaks against the
published `0.0.2` — `ModelSpec` is now structure-only and the `mcpower` feature
is renamed `loop_advanced`. `0.0.3` was never published, so it is not a
migration source.

All four estimators are wired into the stable `fit` dispatch: OLS; GLM
(Gaussian, binomial logit/probit, Poisson, Gamma, negative binomial); LMM
(closed-form single-intercept + BOBYQA general); GLMM (dense and sparse-Z, all
families including NB), with AGQ (nAGQ > 1) for up to 3 random effects per
group (single grouping factor, binomial/Poisson).
Validated against R/lme4 and Julia/MixedModels.jl across a 23-rung dataset
parity manifest plus a 15-rung prior-weights harness.

### Fixed

- **A factor's level order is no longer silently discarded.** `glmm::formula`
  sorted every factor's levels lexicographically, so the treatment-contrast base
  was whichever label sorted first, regardless of what the caller asked for. A
  deliberately ordered categorical — `pd.Categorical(x, categories=["low",
  "med", "high"])` — was refactored to base `"high"`, returning a different
  coefficient for a different question with nothing in the output to reveal it.
  `Column::Factor` now takes `{ levels, codes }`, so the caller states the order
  and level 0 is the base. Python passes a `Categorical`'s
  `categories`/`codes` through; a plain string column has no declared order and
  is sorted by `Column::factor_from_labels` — the same lexicographic default as
  R's `factor()`, now a default rather than an imposition.
- Python: a categorical of non-strings (`pd.Categorical([1, 2, 3])`) was
  classified numeric and fit as one continuous slope instead of expanding to
  dummies. Column classification now checks the dtype before sniffing values.
- Python: `summary()` printed `group 0` instead of the grouping's name, and its
  per-term rows carried no labels — `Lowered::re_groups` was never carried
  across the PyO3 shim. It now is, and `summary()` prints e.g. `Subject:` with
  `(Intercept)`/`Days` rows.

### Added

- `glmm::formula` — the R-style formula frontend is now part of the crate,
  behind the `formula` feature (on by default). `lower("y ~ x + (1|g)", &table,
  family)` builds the kernel's inputs from a formula string and a data table.
  Previously an unpublished companion crate, so it was unreachable for anyone
  installing from crates.io.
- `default-features = false` gives the formula-free kernel, which links no
  `regex` — the configuration for parse-once/fit-many hot paths.
- **`Fit::vcov`** — the full `p×p` fixed-effect covariance `Cov(β̂)`, on every
  path. `Fit::se` is its diagonal and cannot answer anything about two
  coefficients jointly, so a contrast, a confidence interval, or anything of
  `vcov()`/`confint`/`glht`/`emmeans`'s shape needed off-diagonals that were
  being computed and thrown away (GLMM) or never formed (OLS/GLM/LMM). It is
  finite exactly where `se` is. Also on the Python `Fit`, as a `(p, p)` array.
- Python `Fit` gained `n_eval` (optimizer evaluation count), `deviance` (the
  minimized criterion — **not** comparable across models, see the docs), and
  `re_groups`. All three were already on the Rust `Fit`; none crossed the shim.

### Changed

- **Python: `theta=` is renamed `init_theta=`.** One call had two unrelated
  parameters named `theta`: the negative-binomial shape seed and, inside
  `warm_start={"theta": …}`, the random-effect Cholesky vector. The seed takes
  the name R already uses for it (`MASS::glm.nb(init.theta=)`);
  `warm_start["theta"]` is unchanged, matching lme4's `start=list(theta=)`.
- **Python: `targets=` is removed.** It exposed `FitOptions::target_indices`, a
  performance knob for MCPower's hot path that leaves non-target SEs `NaN`. That
  hot path drives the Rust surface directly, where the option is unchanged; no
  Python caller wants `summary()` printing `NA` for standard errors it could
  have computed.
- Python: the native call returns a dict keyed by field name rather than a
  positional tuple. Internal, but it is why `re_groups`/`n_eval`/`deviance`
  could go missing unnoticed.

### LMM cold start

#### Changed

- **LMM cold starts now use the unit-diagonal blind seed** (diagonal θ at 1,
  off-diagonal vech entries at 0 — the lme4/MixedModels convention), on both
  the dense (`fit_lmm`) and sparse (`fit_mle_sparse`) Gaussian paths. The
  former start set *every* component to 1; on wide-slope designs (q ≥ 4 with
  correlated slopes) that start funneled BOBYQA into a second-best local
  optimum on 8 of 9 adjudicated grid cells (deviance gaps +0.23 to +57.4 vs
  the best-known optima, now frozen as goldens under `parity/goldens/optima/`).
  With the new seed the fitted optimum matches or beats MixedModels on all 9.
  Intercept-only and uncorrelated-slope models have no off-diagonal
  components and are bit-identical. Full-grid effect vs MixedModels on the
  gaussian slope stratum: worse-than-MM cells drop 8 → 2 — the two
  remaining are *new* coin-flips where the old start happened to hold the
  best-known basin (`lmm_q6_g300p5_bal_base` +0.008,
  `lmm_q8_g3000p5_bal_lowsnr` +2.03; goldens frozen for both). It also fixes
  the dense-vs-sparse basin disagreement behind the `noz_sparse_grid_agrees`
  cell-20 failure. Eval counts on affected wide-slope fits move both ways
  (grid-wide gaussian-slope total −10%). The sparse non-Gaussian GLMM joint
  seed already used this shape; the Gaussian paths now match it.

### Prior weights

#### Added

- **`FitOptions::weights`** — per-row prior (case) weights, lme4's `weights=`.
  An aggregated binomial (y = success proportion, weight = trial count) now
  fits directly — lme4's `cbind(s, m−s)` objective, which shares its argmin
  (and so β/SE/varcomp) with the expanded-Bernoulli fit — letting the
  `sim_sparse_binomial` parity rung fit its 240 aggregated rows instead of the
  3,059-row Bernoulli expansion. Parity holds at unchanged tolerances; the
  per-solve O(n·width²) cost collapses accordingly.

#### Changed

- `FitOptions.weights` now supported on all paths (was: sparse binomial GLMM
  only); nAGQ>1 with weights rejected.

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
