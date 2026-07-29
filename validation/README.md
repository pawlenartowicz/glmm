# glmm validation suite

Fits the **same statistical model on the same data** with **R `lme4`**, **Julia
`MixedModels.jl`**, and the Rust **`glmm`** crate (plus its Python and R ports),
then compares **estimates** (correctness gate) and **fit time** (informational).
Two independent reference engines agreeing within tolerance is the strong-truth
condition `glmm` is held to.

## The oracle is sacred

> The committed `data/` CSVs and the R / Julia result JSONs regenerated from
> them **are** the frozen oracle. On any disagreement **`glmm` is presumed
> wrong** — investigate and fix `glmm`. A reference result is regenerated
> **only** if the reference *model spec* itself is proven wrong (wrong formula,
> family, link), with a recorded justification. **Never** relax a tolerance or
> edit a reference to make `glmm` pass.

Where the two references disagree with each other beyond tolerance, that is a
recorded **flag to investigate**, never a silent "pick the closer one"
(known exemptions: the GLMM SE-method split and the 2-way rungs listed in the
table). Cross-reference disagreements are catalogued in the workspace's
`docs/parity_gaps.md`.

## Layout

    manifest.json     single source of truth: all 43 rungs, both formula dialects,
                      per-rung options (data, link, weights_col, tier)
    run.sh            the runner -- see Running below
    compare.R         the cross-engine agreement GATE + the two port gates
    tol.R             every tolerance band, with its measurement history
    summarize_timing.R / summarize_accuracy.R / summarize_parallel.R
                      human-readable views (no gate): timing + speedups,
                      accuracy diffs, parallel-feature speedup
    summarize_memory.R
                      human-readable view (no gate) of the memory/ legs
    memory/           peak-RSS harness (no gate): memory.sh measures every
                      engine's peak resident memory over the 43 manifest rungs
                      plus 13 large synthetic models (models.json, regenerable
                      by gen_models.py); one fit script per engine; results
                      land in results/memory/ (gitignored)
    prep/             regenerate data/ from fixed seeds (export_data.R rungs 1-28,
                      gen_weights_data.R rungs 29-43)
    engines/          one fit harness per engine, named for its engine:
                      lme4.R, mixedmodels.jl, glmm.rs (example validation_fit),
                      glmm_python.py, glmm_r.R, common.rs, goldens_agq.R
    data/empirical/   committed CSVs from lme4/nlme-bundled datasets -- EVERY
                      engine reads these bytes
    data/simulated/   committed fixed-seed CSVs (regenerable byte-identically
                      by prep/; committed because cargo test include_str! reads
                      them, incl. in CI)
    goldens/          frozen single-reference results gated in-crate (AGQ tiers,
                      Gamma/NB families MixedModels can't fit, optima/ from the
                      speed-grid adjudication)
    results/          per-engine result JSONs, regenerated every run, gitignored
    campaigns/        three archived studies -- see campaigns/README.md

## What runs where (CI vs local)

| Check | Reads | Runs in CI? |
|---|---|---|
| `cargo test` (4 feature configs) | `goldens/`, `data/empirical/`, `data/simulated/` via `include_str!` | **yes** — every push |
| `cargo test --features oracle-tests` (Tier 2) | `results/` + `data/` on disk | no — local only (needs a prior `./run.sh --oracles`) |
| `./run.sh` (this suite) | everything | no — local only (needs R, Julia, the wheel, fastglmm) |
| campaigns | own manifests/results | no — finished studies, rerun by hand |
| `memory/memory.sh` | `data/`, `memory/models.json` | no — local only, measurement not gate |

## Datasets — the rungs

Rungs 1–28 are the model-shape ladder (empirical + fixed-seed simulated);
rungs 29–43 are the prior-weights tier (`tier: "weights"` in the manifest —
same engines, same gates, plus per-rung weight pathologies).

| Rung | Dataset | Family | What it exercises |
|------|---------|--------|--------------------|
| 1  | Dyestuff | gaussian (intercept) | single-intercept closed-form shortcut |
| 2  | sleepstudy | gaussian (intercept+slope) | correlated intercept+slope RE |
| 3  | Penicillin | gaussian (crossed) | crossed random effects |
| 4  | Pastes | gaussian (nested) | nested random effects |
| 5  | cbpp | binomial GLMM | aggregated binomial (prop + weights) GLMM |
| 6  | grouseticks | Poisson GLMM | Poisson GLMM, 3-way crossed/nested REs |
| 7  | sim_slope_extra | gaussian (crossed slopes, sparse-routed) | slope-carrying extra grouping routed to sparse Z |
| 8  | sim_sparse_binomial | binomial GLMM (7 crossed extras, over-cap) | sparse GLMM over `MAX_EXTRA_GROUPINGS`, binomial |
| 9  | sim_sparse_poisson | Poisson GLMM (7 crossed extras, over-cap) | sparse GLMM over cap, Poisson |
| 10 | Machines | gaussian (q=3 correlated slopes) | wider correlated RE block |
| 11 | Oats | gaussian (real 2-level nesting) | real 2-level nesting |
| 12 | VerbAgg | binomial GLMM (ungrouped 0/1, crossed) | ungrouped binary response, crossed REs |
| 13 | cake | gaussian (`:`-interaction grouping) | `:`-interaction grouping factor |
| 14 | Arabidopsis | Poisson GLMM (real nested) | real nested Poisson GLMM |
| 15 | sim_three_level | gaussian (3-level nesting) | 3-level nesting |
| 16 | sim_max_q_slope | gaussian (q=8, `MAX_PRIMARY_Q` boundary) | RE block exactly at `MAX_PRIMARY_Q` |
| 17 | sim_crossed_at_cap | Poisson GLMM (6 crossed extras, at cap) | sparse GLMM exactly at `MAX_EXTRA_GROUPINGS` |
| 18 | sim_binomial_slope_crossed | binomial GLMM (2 crossed q=2 groupings) | sparse GLMM, two slope-carrying crossed groupings; 2-way gate (MixedModels can't construct) |
| 19 | sim_poisson_nested | Poisson GLMM (3-level nesting) | 3-level nested Poisson GLMM |
| 20 | sim_unbalanced_nested | gaussian (3-level, heavily unbalanced) | heavily unbalanced nesting |
| 21 | sim_nested_crossed_mix | gaussian (nested + crossed combined) | nested + crossed combined in one model |
| 22 | cbpp_probit | binomial GLMM, probit link (cbpp data) | non-canonical link, Fisher-scoring PIRLS branch |
| 23 | sim_gamma | gamma GLMM, log link (dense) | dispersion-family GLMM, dense kernel |
| 24 | sim_sparse_gamma | gamma GLMM, log link (q=5, sparse) | dispersion-family GLMM, sparse kernel |
| 25 | sim_binomial_slope1 | binomial GLMM (Bernoulli, single q=2 grouping) | vector-RE AGQ Laplace anchor |
| 26 | sim_poisson_slope1 | Poisson GLMM (sparse counts, single q=2 grouping) | vector-RE AGQ Laplace anchor |
| 27 | sim_binomial_slope2 | binomial GLMM (Bernoulli, single q=3 grouping) | vector-RE AGQ Laplace anchor, q=3 |
| 28 | sim_poisson_offset | Poisson GLMM, single RE | model offset (`log_exposure`) |
| 29 | wls_basic | gaussian (weighted OLS) | basic prior-weighted OLS |
| 30 | glm_binomial_agg | binomial GLM (aggregated) | aggregated binomial with `size` weights, no RE |
| 31 | glm_poisson | Poisson GLM | weighted Poisson GLM |
| 32 | glm_gamma | gamma GLM | weighted gamma GLM |
| 33 | glm_nb | negative-binomial GLM | weighted NB GLM |
| 34 | lmm_intercept | gaussian (weighted, intercept RE) | weighted LMM, single intercept RE |
| 35 | lmm_slope | gaussian (weighted, slope RE) | weighted LMM, correlated slope RE |
| 36 | lmm_crossed | gaussian (weighted, two crossed slope REs) | weighted LMM, two crossed slope-carrying REs; R-only 2-way gate (pinned MixedModels' `wts=` is broken on this shape) |
| 37 | glmm_poisson | Poisson GLMM (weighted, intercept RE) | weighted Poisson GLMM |
| 38 | glmm_binomial | binomial GLMM (weighted, 8 crossed groupings) | weighted sparse binomial GLMM, `size` weights |
| 39 | path_extreme_range | gaussian (weighted) | weights spanning an extreme dynamic range |
| 40 | path_near_zero | gaussian (weighted) | near-zero weights + the `dropped_rows` gate |
| 41 | path_huge_int | gaussian (weighted, intercept RE) | huge-integer weight values |
| 42 | path_dominant | gaussian (weighted) | one dominant weight overwhelming the rest |
| 43 | all_ones | gaussian (weighted, slope RE) | weights of 1 must equal the unweighted fit bit-for-bit (`unit_identity` gate) |

Status: all 43 rungs green against both references, except the documented 2-way
gates (18, 23, 24 — MixedModels cannot construct or fits a different estimator;
see manifest comments) and rung 36 (weights tier, R-only for a related
MixedModels `wts=` limitation: broken on a slope block combined with a second
crossed grouping). Rung 24 additionally documents R's `as.numeric` 1-ULP parse
quirk (see tol.R / the KNOWN flags below).

Each dataset is exported **once** to neutral CSV; Julia and Rust read that CSV,
not their ecosystem's own copy — this guarantees byte-identical input. The
manifest's `factors` list names the columns every engine re-coerces to
categorical; both references use the first sorted level as contrast base,
asserted via `coef_names`.

## Running

    ./run.sh                    fit glmm (Rust) + Python port + R port, compare vs the
                                EXISTING lme4/MixedModels results on disk (fast default)
    ./run.sh --oracles          refit ALL engines incl. R + Julia (oracle regeneration)
    ./run.sh --prep             regenerate data/ first; implies --oracles
    ./run.sh --rust-tier2       also run the crate's oracle-tests tier first
    ./run.sh cbpp sleepstudy    restrict any of the above to named datasets

Setup (one-time):

**Python** (the port engine) needs the wheel installed **`--release`** into
`../python/venv` — a debug build reports the port's overhead an order of
magnitude too large. `run.sh` prefers that venv and skips the engine if
`import glmm` fails:

```sh
cd ../python && VIRTUAL_ENV="$PWD/venv" venv/bin/maturin develop --release
```

**R** needs `lme4` + `jsonlite` — plus **`GLMMadaptive`** to regenerate the
vector-RE AGQ goldens (`engines/goldens_agq.R` only; ordinary `run.sh` runs,
including `--oracles`, never load it):

```sh
Rscript -e 'install.packages(c("lme4", "jsonlite", "GLMMadaptive"), repos = "https://cloud.r-project.org")'
```

**Julia** runs in this dir's pinned env; set it up once (this writes
`Project.toml` + `Manifest.toml` — commit them for a reproducibly regenerable
oracle):

```sh
julia --project=. -e 'using Pkg; Pkg.add(["MixedModels","CSV","DataFrames","JSON3"]); Pkg.instantiate()'
```

Timing is meaningful only on a locked machine (the **user** runs `bench-l`;
this harness only records lock state). Timing never gates correctness.

## Result schema and SE methods

One file per (dataset × engine), uniform field names so comparison is
mechanical:

```jsonc
{ "dataset": "sleepstudy", "engine": "lme4", "engine_version": "1.1-35",
  "family": "gaussian", "reml": true, "rung": 2, "converged": true, "singular": false,
  "coef_names": ["(Intercept)", "Days"],
  "estimates": {
    "beta": [...], "se": [...], "sigma": 25.59, "loglik": -871.81,
    "varcomp": [ { "group": "Subject", "terms": ["(Intercept)", "Days"],
                  "stddev": [24.74, 5.92], "corr": [[1, 0.07], [0.07, 1]] } ] },
  "timing": { "fit_seconds_median": 0.0123, "n_runs": 100, "warmup_discarded": 1,
              "fits_per_sample": 1 } }
  // GLMM rungs split timing by SE method: `fit_seconds_median` is replaced by
  // `fit_seconds_median_rx` and `fit_seconds_median_hessian` (MixedModels emits only
  // _rx -- no Hessian variant). The Hessian SE (FD-Hessian / numDeriv) is the main
  // time consumer; on cbpp glmm's Hessian fit is ~1.9x its Rx fit.
```

The single `se` above is a **gaussian (LMM) rung** — its SE is profiled, one
method. On **GLMM rungs** the Laplace SE has two genuinely different
variants, so `se` is replaced by **`se_hessian`** (keeps the θ–β coupling)
and **`se_rx`** (conditional on θ̂, drops it); MixedModels computes only
`se_rx`, so its GLMM files carry that field alone. This is what lets
`compare.R` measure like method against like.

Both fit scripts normalize the careful bits to one representation:
**`varcomp`** to the absolute σ scale (lme4 reports variances, MixedModels
σ-relative θ); **`loglik`** to the logLik scale (MixedModels reports the
−2·logLik objective); **`reml`** is honored from the manifest, not just
recorded (MixedModels defaults to ML, which biases varcomp downward — `null`
and `sigma` absent on GLMM rungs).

## Ports

`glmm_python` and `glmm_r` are **not independent implementations**: both reach
the identical Rust kernel (PyO3 / extendr). compare.R therefore gates them
against the **Rust row** at `TOL$port_rel` (a round-off band, measured 0 on
every gated quantity) — a miss means the port fed the kernel different input
(swapped column, mis-sorted factor levels, dropped weights), exactly the bug
class the ports' own test suites cannot see.

`compare.R` also carries a `KNOWN_R_PARSE` mechanism (currently
`sim_max_q_slope` and `sim_binomial_slope2`) that relabels a port-gate FAIL to
`KNOWN` instead of counting it toward the pass/fail verdict, for the
documented case where R's `as.numeric` is not correctly rounded on some
decimal strings (1 ULP off Rust/Python) and that shift selects a neighbouring
optimum on those two multimodal correlated-slope surfaces. As of the last
`--oracles` run this mechanism did not trigger — both rungs gated clean at
`TOL$port_rel` like every other rung — but the relabelling logic stays in
`compare.R` for the numerical-limit case it exists to handle. `TOL$port_rel`
is never relaxed.

Timing across engines is same-to-same: every engine times the
construct-and-fit call; the Rust harness also records a fit-only median for
solver work. `summarize_timing.R` reports per-engine medians, speedups vs
lme4/MixedModels, and the port-call overhead columns (`py_gap`, `r_gap`).

## Tolerances

Per-quantity bands live in `tol.R`, one place, each with the measurement that
set it and its history. Never relaxed to make an engine pass.

## Estimator pinning

The cross-engine sweep is pinned to Laplace (lme4 `nAGQ=1`) — like-for-like
with glmm's glmer-faithful kernel. AGQ validation (`nAGQ` 7/11, scalar and
vector RE) lives in the **goldens track**: `engines/goldens_agq.R` freezes
lme4/GLMMadaptive references into `goldens/`, gated in-crate
(`src/fit/glmm_tests.rs`) at the `agq_*` bands in `tol.R`.
