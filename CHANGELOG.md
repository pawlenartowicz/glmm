# Changelog

All notable changes to the `glmm` crate are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The Python package (`glmm` on PyPI) is versioned in lockstep with the crate and
shares these entries; Python-specific notes are called out where they differ.

## [0.2.0] — unreleased

A correctness release that breaks one thing on purpose. Near-collinearity is a
spectrum, and the crate treated most of it as failure: a design whose columns
were merely hard to separate came back all-NaN or quietly one column short, and
on the weighted-OLS route it came back wrong. The rank guards measured
`min|L_ii| / max|L_ii|`, which tracks column *scale*, not near-dependence. They
now measure the scale-invariant per-column pivot ratio, and below the floor the
dense LMM, OLS and GLM routes **fit and flag** — real coefficients, a large
standard error, a machine-readable note naming the column — rather than refuse.
Sparse LMM still refuses. Genuine redundancy is untouched: the alias gate keeps
dropping exactly-dependent columns at `ALIAS_EPS`, bit-identically.

The break is the diagnostics consolidation riding with it: `converged`,
`singular` and `aliased` — plus two internal channels that never reached the
result and the new ill-conditioned marker — now sit behind one
`fit.diagnostics`, and `Fit` plus the three new types become `#[non_exhaustive]`
in the same change, so the next diagnostic is additive instead of the next
break. Rust callers change field access to an accessor or one extra hop; Python
and R callers see additions only.

### Changed

- Raised the sparse PIRLS step-halving cap (`PIRLS_MAX_HALVINGS`) from 10 to
  16. The sparse GLMM's Wald-Hessian standard-error step cold-starts its
  finite-difference deviance evaluations from `û = 0`, and on a large-θ̂,
  many-crossed-grouping design that cold start needed one more halving than
  the cap allowed to walk back to the mode, hard-failing a fit that had
  otherwise converged cleanly. The cap now carries margin above the measured
  floor; no other PIRLS behavior changed, and no existing validation result
  moved (full corpus re-fit, bit-identical).

- **BREAKING — `converged`, `singular` and `aliased` moved off `Fit` into
  `fit.diagnostics`.** `Fit::converged()`, `Fit::singular()` and
  `Fit::aliased()` forward to them, so most call sites change by one character
  or not at all; what breaks is field access, struct literals and exhaustive
  destructuring. One storage location — the accessors read the same
  `Diagnostics`. `Fit::aliased()` returns `&[bool]`; ownership is
  `fit.diagnostics.aliased`.

- **BREAKING — `Fit`, `Diagnostics`, `Boundary` and `Note` are
  `#[non_exhaustive]`.** A one-time break so every future diagnostic, boundary
  state or note variant is additive. `Fit` is no longer constructible outside
  the crate.

- **The rank guards measure a different statistic, and refusal became
  flagging.** The condemning control: `y ~ 1 + u + w + (1|g)` with nothing
  collinear, varying only one column's scale — the old statistic fell six
  decades (5.4e-8 → 5.4e-14) while β̂ moved less than one part in 1e10, and
  three of those four fits were discarded. Every guard now uses
  `min_pivot_ratio` (per-column Schur pivot ÷ that column's own Gram diagonal,
  the alias gate's own basis). Floors: dense LMM and OLS/GLM flag below
  `1e-12`, sparse LMM refuses below `6e-10`. The `EPS_RANK` reveal-and-retry
  that dropped a column after the solver tripped is gone with its constant; the
  alias gate (`detect_aliased` on the raw `X'X` at `ALIAS_EPS`) is now the only
  place a column is ever dropped. `chol_rank_deficient` survives narrowed to
  `src/lme.rs`'s private Brent kernel, which fits off sufficient statistics and
  has no design in hand to take a pivot from.

- **The FD-Hessian θ step is absolute, not SD-scaled** (dense GLMM
  `se_hessian`). `fd_hessian_cov` stepped every joint coordinate at
  `FD_STEP_REL · max(1, |γ̂_k|)` — right for β, backwards for θ, where a larger
  random-effect SD flattens the profile and the rule let the O(h²) truncation
  error grow as θ̂². θ now steps absolutely (constant renamed
  `FD_STEP_BASE`, value unchanged at 1e-2); β keeps the relative form. Eleven
  fits moved, all dense GLMM `se_hessian` or θ-block SEs off the same Hessian,
  each named in advance and re-pinned with provenance; fits with
  `max(1, |θ̂|) = 1` are bit-identical. Against `glmer`
  `vcov(use.hessian = TRUE)` at `tolPwrss = 1e-13`: 3.9e-5 → 1.4e-5 (θ̂ = 2.97)
  and 1.3e-5 → 7.4e-6 (θ̂ = 4.51). The residual gap is reference-limited —
  glmm sits within 3.1e-6 of its own h→0 limit, while lme4's value is itself a
  δ = 1e-4 finite difference carrying 5–9e-6.

- **The singular-fit warning stops asserting things the numbers contradict.**
  Both ports said `sd(term | group) = 0`; they now say `pinned at the variance
  boundary` — the pin fixes the Cholesky *diagonal*, so a q ≥ 2 block's
  reported stddev keeps the off-diagonal (measured 2.2e-3, which prints in a
  `VarCorr`). The `corr(a, b | group) = ±1` clause is removed from both ports:
  a q ≥ 2 pin *is* that event and `pinned` reports it reliably, while the old
  exact `abs(corr) == 1` test never fired (measured 1.0000000000000002). The
  ill-conditioned message likewise says "*b* is entangled with one or more
  other columns" — entanglement is symmetric and the kernel names whichever
  column its pivot search reached. See the corrected 0.1.1 entry below.

- **The NB θ-search stopping width is `1e-4` on `ln θ`, was `1e-8`**
  (`golden_max_ln_theta`, shared by the GLM conditional θ profile and the
  GLMM marginal-θ route). Every evaluation the search makes is a full inner
  fit converged only to its own noise floor (`GLMM_RHO_END = 3e-6`), and the
  old width asked for four decades more resolution than that inner fit
  could supply — the last ~13 of ~45 iterations were picking a side of a
  knife-edge in the inner fit's own basin selection, not resolving curvature.
  Traced on the dense NB GLMM fixture: a 1-ULP input perturbation used to
  flip the reported β by 9.5e-5 relative through exactly this mechanism; at
  the new width the same perturbation converges to a bit-identical β. Moves
  β/θ̂/varcorr on every NB fit that reaches the search (dense and sparse
  GLMM, the fixed-effects GLM θ profile) by amounts inside their existing
  lme4-facing bands; no cross-engine golden moved.

- **`bobyqa` bumped `0.1.3` → `0.2.0`** — the optimizer every LMM and GLMM
  θ-search runs on. The crate's own call sites are unchanged: `Config`,
  `Bobyqa`, `RestartConfig`, `Status` and `Outcome` are used exactly as
  before.

### Added

- **`Diagnostics`, `Boundary` and `Note`**, re-exported from the crate root.
  `Diagnostics` carries `converged`, `singular`, `aliased`, `boundary`,
  `pinned`, `notes`. `Boundary` is `Interior` / `AtBoundary` / `NoOptimum` —
  the last is the one fact nothing exposed before (optimizer cap-out,
  previously inferable only from a finite `deviance` with `converged` false).
  `Note` has three variants. `IllConditioned { columns, pivot }` carries the
  measured pivot so callers can rank severity.
  `PirlsExhausted { evals, final_eval }` says a GLMM's inner PIRLS solve ran
  the full 50-iteration cap without meeting its band; `final_eval` separates
  the case that matters (the solve behind the reported estimates) from a
  rejected BOBYQA trial point. `UnusedGroupingLevels { grouping, levels }` is
  raised by the formula frontend, not a solver, and names declared levels that
  carry no row but still occupy random-effect columns. A fourth, anticipated
  variant was dropped: measurement found no regime where β is noise *and* the
  standard error lies about it, so there is nothing for a refusal to protect.

- **`Diagnostics::pinned`** — which variance components collapsed, as flags
  aligned with the `varcorr` blocks (`pinned[g][i]` pairs with
  `stddev_corr(g).0[i]`), replacing an internal bitmask keyed to a non-public
  order. Empty means "nothing to report"; `AtBoundary` with empty `pinned`
  means something pinned on a route that cannot say which.

- **Coverage is documented rather than assumed.** `converged` and `aliased`
  are filled on every route; `boundary` and `pinned` are real wherever
  variance components exist; `notes` is per-variant. `IllConditioned` can only
  be raised by OLS, GLM and dense LMM — dense GLMM records no pivot, sparse
  refuses instead of flagging. `PirlsExhausted` comes from the GLMM routes,
  dense and sparse. `UnusedGroupingLevels` comes from the formula frontend
  (`formula::Lowered::notes`), so a caller building `x`/`ModelSpec` by hand
  never sees it. An absent note means "not detected", never "checked and
  clean".

- **Python: `fit.diagnostics`** (dict, same six keys) plus one warning class
  per note variant — `glmm.DiagnosticWarning` (base, a `UserWarning`),
  `glmm.IllConditionedWarning`, `glmm.PirlsExhaustedWarning` and
  `glmm.UnusedGroupingLevelsWarning`. Filter the whole channel or one variant
  with `warnings.filterwarnings`. `columns` are 0-based indices into
  `Fit.names`.

- **R: `fit$diagnostics`** (list, same six names); every existing top-level
  name is unchanged and `isSingular()` is unaffected. Notes arrive as classed
  conditions — `fastglmm_ill_conditioned`, `fastglmm_pirls_exhausted`,
  `fastglmm_unused_grouping_levels` and `fastglmm_unknown_note` (the forward
  compatibility arm for a variant this version of the port does not know),
  all inheriting `fastglmm_diagnostic` — selectable without matching message
  text. `columns` are 1-based on this side.

- **The Gaussian LMM paths now report `ranef` and `fitted`.** Both were empty
  there since 0.1.1 — those paths fit off sufficient statistics and never
  formed per-row quantities. The conditional modes are now recovered at θ̂
  after the fit, and the fitted means built from them (offset restored), so
  `Fit::ranef`/`Fit::ranef_levels`/`Fit::fitted` are filled on every converged
  route. Checked against a brute-force solve of the penalized normal equations
  that shares no code with the recovery (`tests/lmm_ranef.rs`). This is what
  the R `ranef()`/`fitted()` methods and Python `ranef_blocks` rest on for an
  LMM.

- **Labelled conditional modes.** The kernel has carried `ranef` and
  `ranef_levels` as flat numbers since 0.1.1, but the layout a grouping lands
  in is a data-dependent decision inside the kernel, so slicing them by hand
  was never safe. `formula::label_ranef` resolves those numbers back to
  `RanefBlock`s — grouping name, term names, level labels, values — using the
  per-slot labels the lowering now keeps (`ReGroupInfo::slot_labels`), and
  drops a nested grouping's padded slots so `levels` is exactly the levels that
  exist. It returns the new `formula::Error::RanefShapeMismatch` rather than
  panicking when a `Fit` and a lowering do not belong together.

- **Python `Fit.ranef_blocks`** — the same labelled form, a list of dicts with
  `group` / `terms` / `levels` / `values`. Not a DataFrame: NumPy is the
  package's only dependency. Empty exactly when `ranef` is.

- **R `ranef()` and `fitted()` work.** Both were hard errors. `ranef()` returns
  lme4's shape — a named list of data frames, one per grouping, rows labelled
  by level and columns by term. `fitted()` returns the conditional means μ̂ per
  row of the model frame, named by its row names, including the random-effect
  contribution and any offset. `predict()` and `residuals()` still error with
  their reasons; `residuals()`' reason is that lme4's `type` argument picks
  between four different quantities and guessing would silently disagree.

- **The `orchestrate` cargo feature** (off by default) — the string-typed fit
  orchestration both FFI ports need: one definition of the family/link string
  vocabulary, the formula-and-data lowering, and the flattened result they
  publish, plus the panic-to-`Err` boundary. `glmm-python` and `glmm-r` each
  carried a mirrored copy (`orchestrate.rs` + `convert.rs`, ~500 lines apiece);
  both copies are deleted and both ports now call `glmm::orchestrate`. Like
  `loop_advanced` and for the same reason, this is **not** part of the
  semver-covered surface — its shape follows the ports' needs.

- **Two frozen lme4 references for designs the crate used to discard.**
  `validation/prep/gen_illcond_data.R` emits `sim_dynrange_lmm` and
  `sim_entangled_pair_lmm`, bit-identical to the crate's builders, CSVs exact
  at 17 digits. The first is a registered cross-engine golden (β within
  6e-11); the second bands the entangled pair itself — 5.7e-4 on the two
  entangled coefficients, 4.8e-7 on their identified sum, against 1e-3.

- **Large-θ̂ validation coverage.** Every GLMM rung's RE SD sat in
  [0.34, 1.34], so the suite had never seen the Laplace approximation's weak
  regime. Three committed datasets — `sim_binomial_bigsd` (θ̂ = 4.51),
  `sim_poisson_bigsd` (θ̂ = 2.97), `sim_binomial_zerosd` (θ̂ exactly 0) — plus
  two frozen lme4 rung references, two AGQ goldens at nAGQ = 7 and 11, a
  validated per-rung tighten-only `se_hessian` band (the two rungs gate at
  3e-5, where the default 1e-3 would have caught nothing), and an in-crate
  convergence-flag assertion at the θ → 0 boundary, where lme4 says
  `isSingular` and glmm says `converged = true, singular = true` — a real
  divergence `compare.R` cannot express.

- **Runnable examples and a documentation index.** `documentation/index.md`
  plus `examples-python.md` / `examples-r.md` and the nine paired scripts each
  walks through (`documentation/examples/{python,r}/`), which check themselves
  against lme4 values via a small oracle helper, and
  `coming-from-statsmodels.md`. The three tutorials moved to lowercase
  filenames (`tutorial-python.md`, `tutorial-r.md`, `tutorial-rust.md`).

- **`logLik()`, `AIC()` and `BIC()` work in the R package.** The kernel has
  carried `loglik`/`df`/`reml` since 0.1.1 and Python exposed them; the R fit
  object dropped all three on the way out behind a stale error. `logLik()` now
  returns a `"logLik"` object with `df`, `nobs`, `REML` attributes. On an LMM
  the value is the REML criterion (`REML = TRUE`), comparable only across
  identical fixed effects — which is why `summary()` still prints no
  `AIC BIC logLik` line. R-side plumbing only.

- **`offset=` works in the R package.** Rejected with a message whose stated
  reason (`the kernel has no offset field`) stopped being true in 0.1.1.
  `fastglmm(..., offset =)` now takes a per-row additive term on the
  linear-predictor scale, following `weights=` at every site: evaluated in
  `data` (so `offset = log(exposure)` works) and parked in the model frame so
  `subset=`/`na.action` drop entries row-locked. The `offset()` *formula term*
  is still an error, now saying only that. The new formal sits after
  `na.action` (where `stats::glm` puts it), so a call passing `nAGQ` or later
  arguments positionally shifts by one. The R port now fits the offset rung,
  covered by the port gate at 9e-16.

### Fixed

- **Weighted OLS returned silently wrong coefficients and called them
  converged.** On a design full-rank on the raw `x` but singular once prior
  weights apply, the old guard's statistic bottomed out four orders *above*
  its own threshold even where `X'WX` was numerically indefinite, and the
  pre-dispatch alias gate tests the raw `x` (healthy at every rung). The crate
  returned `β = −1527` for a true 0.477 with `converged: true`. The root cause
  was the statistic, not the threshold. Such a design now fits with an SE more
  than 100× its coefficient and an `IllConditioned` note.

- **The GLM divergence guard rejected well-behaved fits in the wrong units.**
  It bounded `|β_j| > 30` at iteration ≥ 3, which is not a property of the
  model: rescaling a predictor column rescales β̂ exactly, so the same fit was
  accepted or refused depending on the caller's units. `y ~ x/1000` came back
  `converged: false` with β̂ = (0.4915, 803.09) and NaN standard errors while
  the identical model at unit scale converged, on both the logit and the
  Poisson log link. The bound is now on the linear predictor
  (`ETA_DIVERGENCE_CAP`, same value 30), which `η = Xβ` leaves unchanged under
  a unit change — and where 30 is the number the argument supports (on the
  logit scale `|η| = 30` is p ≈ 1 − 1e-13, separation rather than signal). The
  guard is skipped under the Gamma inverse link, where `η = 1/μ` makes a large
  `|η|` an honest small-mean fit; that pairing falls through to `clamp_eta`,
  the non-finite β check and the iteration cap as before. `BETA_BOX` (the
  joint BOBYQA's β box) was aliased to the old cap and is now its own
  constant, unchanged at 30, so the two stop moving together by accident.

- **A pinned variance component on a q ≥ 2 grouping went unreported in both
  ports.** The ports reconstructed the pin set by scanning `varcorr` for exact
  zeros — which a q ≥ 2 pin never produces (stddev keeps the off-diagonal,
  measured 2.2e-10; the corr fallback missed at 1.0000000000000002), so the
  user got the bare lme4 text. Both reconstructions are deleted; the ports
  read `Diagnostics::pinned`, the kernel's own record.

- **Sparse fits now name their collapsed components.** Both sparse routes ran
  a per-component pin loop but recorded one "something pinned" bool; they now
  build the same mask the dense routes do. Diagnostics only — no fitted number
  moved.

- **A subnormal response punched a `-inf` hole in the Poisson and NB
  objective.** Both deviance residuals computed `y·ln(y/μ)`; for subnormal `y`
  and `μ ≥ 2` the quotient underflows to exactly `0.0` before the log, so the
  term evaluated to `-inf` where the true value is finite and tiny. Written
  `y·(ln y − ln μ)` instead, which is the same quantity with each log taken on
  a representable argument (`src/family.rs`). Only reachable with a response
  at the bottom of the exponent range; no fitted number on any normal-range
  data moved.

- **An aliased column used as a random slope returns a fit instead of
  panicking.** `remap_spec_slopes` asserted; it now returns a non-converged
  `Fit` (NaN β/se). Actually fitting the reduced model — dropping the random
  slope with its aliased fixed column — is a different model with its own
  oracle, and is not implemented here.

### Removed

- **`LmeScratch::ols_scratch`** (`loop_advanced`, RULE 1): provisioned for a
  τ̂ ≈ 0 OLS fallback never implemented — the boundary case pins θ and runs the
  same profiled-deviance path. Written and read by nobody.

### Changed — `loop_advanced` (no semver guarantee, RULE 1)

Every changed item, named because MCPower consumes this tier.

- `FitView::diagnostics() -> FitDiagnostics` **added** — a `Copy` struct read
  off borrowed state, no per-draw allocation.
- `FitView::boundary_hit()` / `FitView::pinned_components()` **removed**; read
  them off `diagnostics()`. `converged()` is unchanged and forwards there.
- `fit_suff_stats_t_sq` **lost its `eps_rank` parameter** — once OLS stopped
  refusing it guarded nothing.
- **The dense LMM route lost its rank guard at the θ-search endpoint** (the
  `EPS_RANK = 1e-8` test on `X'V⁻¹X` at θ̂ that NaN-filled the fit); it fits
  and flags there instead. `fit_on` drives this route — see Migration. The
  third copy of the predicate, on `src/lme.rs`'s Brent kernel, is untouched.
- **`pinned_components` is always `0` on the sparse and NB (`Prebuilt`)
  draws** — indistinguishable from "nothing pinned". Those routes do fill
  `Diagnostics::pinned` on the assembled `Fit`; the `Copy` carrier holds no
  `Vec`. Read the cold surface when per-component pinning matters.
- `GlmFitView`, `OlsFitView`, `LmmFit` **gained `pivot` / `pivot_col`**.
- `LmmFit::eps_rank_aliased` **removed** with the reveal-and-retry gate.

### Migration

- **Rust:** `fit.converged` → `fit.converged()` or
  `fit.diagnostics.converged`; same for `singular`/`aliased`. `Fit` literals
  and exhaustive matches need rewriting; `..` covers the match case.
- **Python — one real removal, sharper than it looks.** Attribute reads are
  unchanged (`fit.converged` etc. remain as properties), but the three left
  the dataclass **field list**: `glmm.Fit(converged=...)` no longer
  constructs, and `dataclasses.asdict(fit)` silently stops carrying the three
  most-read diagnostics — nothing raises, the keys are simply absent. Code
  round-tripping a `Fit` through `asdict` must read `fit.diagnostics`.
- **R:** nothing breaks; every current name reads the same value.
- **Loop-tier callers: draws that used to arrive as NaN now arrive as
  numbers, on every dense route including OLS. Aggregate without checking the
  flag and your results move.** Three causes: the alias gate runs in
  `fit_warm`, so `fit_on` bypasses it and a rank-deficient draw reaches the
  solver whole (NaN vs fitted-and-flagged is then settled by the arithmetic on
  that draw, not predictable from the design); `fit_suff_stats_t_sq` lost its
  non-scale-invariant guard, so full-rank draws it wrongly discarded (a merely
  rescaled column crossing 1e-12) now come back; and the dense LMM endpoint
  guard is gone, at a threshold four decades looser, so badly-scaled LMM draws
  start moving at column scale 1e-10, not 1e-12. Differentially measured (6
  sizes × 3 seeds, old and new trees in one binary): duplicate-column LMM
  draws that arrived NaN now arrive fitted on 16/18 designs (11/18 with the
  duplicate rescaled); nothing anywhere moved fitted → NaN. Screening is the
  caller's job: read `FitView::diagnostics()` on every draw and check
  `converged` **and** `ill_conditioned` before it enters an aggregate
  (`pivot_col` names the column). A filter dropping only non-converged draws
  silently admits the newly fitted ones.

### Not changed — measured null result

- **Sparse FD-Hessian seeding was implemented on a scratch tree and
  rejected.** The dense path's cold-start defect does not fire on the sparse
  arm (cold `f0` reproduces the converged deviance to ≤7.9e-7 on eight
  fixtures, non-vacuity proven by injection), and at the shipped
  `SPARSE_FD_STEP_REL = 1e-4` the warm seed is actively harmful — PIRLS's
  relative-increment exit trips early, moving `sim_sparse_gamma` `se_hessian`
  by −27% and `sim_sparse_nb` by −61%. Step-coupled (works at h = 1e-2, fails
  at h ≤ 1e-3); blocked on a sparse step recalibration. No code landed;
  `src/sparse/glmm.rs` deliberately keeps its `max(1, |θ̂|)` scaling, which is
  calibrated on the noise side there.

### Internal — test pins

- Seven in-crate pins moved off the flat `PIN_REL_ITER` onto per-test bands
  (5e-6 to 3e-3) sized from measured aarch64-apple-darwin drift; on the
  reference machine every one is bit-exact across all four feature configs.
  `assert_pinned` names the machine and holds the account.
- The `alloc-tests` bounded-allocation tests serialize themselves through
  `test_support::alloc_test_guard` (dhat counts process-wide), so they no
  longer need `--test-threads=1`.
- The `pending_reference` golden-exclusion flag's dead paths are now driven by
  synthetic specs; proven non-vacuous by inverting the predicate.
- **`fit_glmm_nb_sim_matches_lme4` and `fit_glmm_nb_nested_unbalanced_matches_lme4`
  gained an additive bit-exact Rust-vs-Rust pin** (`BAND = 1e-7`, alongside their
  existing lme4 bands): entry 9 found neither dense NB test could tell a
  regression from rounding, since their only assertions were lme4-facing
  bands wide enough to absorb the old θ-search instability. Both fixtures
  clear a 1e-5 conditioning gate by 3+ orders of magnitude at the new
  stopping width, under both a 1-ULP sweep and the lane-width probe.
  **`fit_sparse_nb_glmm_is_pinned` dropped its bit-exact pin** (the crate's
  thinnest-ever margin, on its worst-conditioned NB fit) for oracle
  agreement against the frozen `sim_sparse_nb.json` golden instead — the
  sparse route's own coverage comes from two live both-paths cross-checks
  against dense, not a second frozen-Rust value.

## [0.1.3] — 2026-07-29

An allocation release. Nothing moves an answer and nothing on the public
surface changes shape: the full validation manifest refits bit-identically
before and after every change. Two changes — the dense GLMM workspace stops
allocating large matrices its common routes never read, and the
`loop_advanced` build-once/fit-many tier stops allocating per draw — plus the
memory harness that measured them.

### Changed

- **The dense GLMM `Z` matrices are allocated only on the route that reads
  them.** The dense GLMM workspace allocated three `n × k_total` matrices
  (`Z`, `M`, `WM`) unconditionally, but the two common solve routes never
  read `Z`: the blocked route reconstructs per-cluster blocks on the fly,
  and the structured route needed it only inside `build_packed_m`, which now
  builds its packed products directly from the column structure. The
  workspace now sizes those buffers to 0×0 unless the model routes to the
  dense fallback (the one route that genuinely reads all of them, unchanged).
  Measured peak RSS on large models, Rust binary: a 50,000-row random-intercept
  fit with 800 levels drops 952 → 27 MB, an observation-level-RE fit
  (10,000 rows, 10,000 levels) 3827 → 12 MB, four correlated slopes plus a
  crossed grouping 2765 → 51 MB; on the validation manifest the multi-grouping
  rungs shrink the same way (VerbAgg 54 → 16 MB) and everything else is flat.
  Net of each runtime's baseline, the kernel's fit cost on the blocked shape
  is now below lme4's.

- **The `fit_on` loop tier no longer allocates per draw** (`loop_advanced`
  feature). The `Ols`, `Glm` and dense-LMM arms each built a fresh
  column-major `n × p` copy of `x` on every call; they now fill a buffer
  preallocated once in `build_workspace`, the same pattern the dense-GLMM arm
  already used. The LMM offset path likewise reuses a preallocated `y − o`
  buffer, and unweighted OLS workspaces reclaim the `scaled_x` matrix they
  never touch. Per-draw heap traffic on those arms is gone. Measured on a
  locked clock (pinned P-core, min over repeats, both versions producing
  bit-identical estimates): 2–3 % per draw on OLS, ≤1 % on dense LMM, flat
  on GLM (IRLS iteration cost dwarfs one allocation), at n = 1000–10000.
  An allocation-hygiene change, not a speedup headline.

### Added

- `validation/memory/` — a peak-RSS measurement harness over the 43 manifest
  rungs plus 13 large synthetic models, with cross-engine baselines (Rust
  binary, Python and R ports, lme4, MixedModels.jl) and a summariser
  (`validation/summarize_memory.R`). Measurement tooling only; no gate, no
  golden.

## [0.1.2] — 2026-07-22

A fix release with an internal restructure. Two changes move an answer, both
narrow: the Gamma inverse-link PIRLS boundary fix (a cell that reported false
convergence now lands on lme4's optimum) and the formula random-effect ordering
fix (a formula that writes a plain-intercept term before a slope term now takes
the written order as its primary grouping). Everything else — the whole log-link
and binomial corpus, single-RE and slope-first formulas — refits bit-identically,
and the oracle goldens hold at their existing tolerances.

### Changed

- Restructured the validation suite: `parity/` is now `validation/` (package
  `validation`, example `validation_fit`, test file `tests/validation_oracle.rs`);
  the prior-weights suite merged in as manifest rungs 29–43 (`tier: "weights"`);
  the finished grid/diligent/accuracy studies archived under
  `validation/campaigns/{speed-grid,estimate-grid,monte_carlo}/`. No gate,
  tolerance, golden, or dataset changed.

- **The formula frontend now lowers random effects in formula order.** The
  parser used to emit random effects in its internal extraction order (slope
  terms before plain intercept terms), so in a formula like
  `y ~ x + (1|g) + (1+x|h)` the slope grouping `h` became the primary grouping
  even though `g` was written first. Random effects now follow the order they
  are written, with one exception: a nested `(1|A/B)` pair still sorts first,
  because the kernel interprets nesting relative to the primary grouping. For
  formulas that write a plain-intercept term before a slope term this changes
  the primary grouping, and with it: which solver the model routes to (a slope
  on an extra grouping is a sparse-routing trigger), the packing order of the
  θ variance components, and the order of `ReGroupInfo` blocks. Single-RE
  models, all-intercept models, slope-first models, and anything with a nested
  term lower exactly as before. This fix re-landed parity rung 24
  (`sim_sparse_gamma`) at unchanged tolerances: the misordering had routed it
  to the dense kernel, whose optimizer stops ~2e-3 deviance short on that
  shape, while the formula-order orientation routes sparse and lands 2e-4
  from lme4's optimum.

### Fixed

- **Gamma inverse-link PIRLS could converge on the η > 0 domain boundary.**
  `clamp_eta` projects a trial iterate with η ≤ 0 (where μ = 1/η is undefined)
  onto η = 1e-10, and the projected row's working weight μ² ≈ 1e20 then
  dominates the WLS solve, so PIRLS kept returning the boundary and reported
  convergence there. Routed through the sparse solver, the `sim_gamma`
  inverse-link cell returned `converged = true` at an optimum ~937 deviance
  units above lme4's; the same mechanism put a ~98-unit discontinuity in the θ
  surface BOBYQA minimizes on the dense path, which had been reaching the right
  optimum only because its warm-start chain stayed feasible. All four PIRLS
  drivers now treat a domain-infeasible trial iterate as a failed step and
  halve toward the last accepted feasible iterate (R `glm.fit`'s
  `valideta`-style step-halving); a first trial with no accepted predecessor
  backtracks toward the u = 0 seed, and an infeasible η_fixed itself surfaces
  as an honest non-converged NaN. Every family/link whose η domain is all of ℝ
  — the whole log-link and binomial corpus — refits bit-identically
  before/after the change; the repaired sparse cell is pinned against the dense
  fit and the frozen lme4 golden (`sim_gamma_inv_glmm`).

- **`Sizing::n_clusters_at` under-counted clusters off-grid.** Under
  `Sizing::FixedSize` it divided `n / cluster_size` rounding down, while its
  neighbour `Sizing::cluster_of_row` sends row `i` to cluster `i / cluster_size`.
  With `n = 18, cluster_size = 4` row 17 lands in cluster 4, so five clusters
  exist and the function reported four — the trailing partial cluster is real
  and its id must be in range. It now rounds up, matching the workspace
  allocator, which had been carrying its own private copy of the corrected
  formula. Off-grid `n` only; on an atom multiple the two agree, so no shipped
  path changes answer.

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

  > **Corrected 2026-08-01, and both halves of the claim were wrong.** The
  > `corr(a, b | group) = ±1` clause no longer exists in either port — see the
  > 0.2.0 entry above — and the exactness argument it rested on was never
  > true. What the kernel pins is the Cholesky diagonal, not the reported
  > standard deviation or correlation: on a q ≥ 2 block the stddev keeps the
  > off-diagonal and lands at ~1e-10, and the correlation was measured at
  > 1.0000000000000002. The `corr(...)` clause's exact comparison therefore
  > never fired on the designs it was written for, which is why it was deleted
  > rather than reworded. The `sd(...)` half did ship and does fire — its
  > scan-for-zero catches a q = 1 pin, where the pinned value really is exact
  > 0.0, and misses a q ≥ 2 one for the same off-diagonal reason; that is the
  > bug fixed in the 0.2.0 entry, along with the wording.

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
