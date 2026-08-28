# glmm validation suite

Fits the **same statistical model on the same data** with **R `lme4`**, **Julia
`MixedModels.jl`**, and the Rust **`glmm`** crate (plus its Python and R ports),
then compares **estimates** (a reference check, see below) and **fit time**
(informational). Two independent reference engines agreeing within tolerance is
the strong-truth condition `glmm` is measured against.

## The references are frozen; the deviance gate is hard, parameters are sanity checks

> The committed `data/` CSVs and the R / Julia result JSONs regenerated from them
> **are** the frozen references, and they never move to accommodate `glmm`. A
> reference result is regenerated **only** if the reference *model spec* itself is
> proven wrong (wrong formula, family, link), with a recorded justification.
> **Never** relax a tolerance or edit a reference to make `glmm` pass.

`glmm` is its own best reference: both engines provably minimize the same
objective, so the converged **deviance** (`dev = -2 * loglik`, each engine's own
reported `loglik`) is what actually distinguishes a real regression from a
different point on a flat surface. That is where the hard gate lives, on the
`glmm`-vs-lme4 row only:

- **worse converged deviance than lme4 is a hard fail** — `Δdev = dev_glmm -
  dev_lme4 > TOL$dev_eps` — however the parameters look;
- **`Δdev <= 0` always passes** (`DEV-WIN`, glmm found an equal-or-better optimum
  of the shared objective) and is listed in the run summary, not silent;
- **`|Δdev| > TOL$dev_big`, either direction, is `FAIL(conv?)`** — a suspected
  deviance-convention mismatch (e.g. a missing saturated-model constant), not a
  fit result, and both raw deviances print so the mismatch is legible;
- a rung with no usable reference `loglik` is `DEV-NA(why)` — a loud exclusion,
  listed in the run summary, and never a silent pass.

`TOL$dev_eps` and `TOL$dev_big` are pinned in `tol.R` from a corpus-wide Δdev
floor measurement, not from taste — `measure_dev_floor.R` produces the table, and
`tol.R`'s own comment carries the resulting numbers and the measurement date.
Deviance is only comparable after per-family,
per-method convention alignment — `dev_align.R` does that (Gaussian REML
constant, lme4 nAGQ>1's missing saturated-model term, NA-with-why when a
reference has no usable `loglik`) and is the single place that inventory lives,
mirrored by the crate's own `tests/oracle_support` dev_align module for the
in-crate tier. The glmm-vs-MixedModels.jl Δdev is computed and printed
informationally; the hard gate runs against lme4 only.

Parameter comparisons (β, SEs, varcorr, stddev) are **sanity checks**, not a
pass/fail gate on their own: a quantity outside its band passes **only** when
`divergences.json` carries an entry for it — see
[Documented divergences](#documented-divergences). Anything outside its band
with no entry still fails the run, exactly as before; what changed is that an
out-of-band parameter with an equal-or-better deviance is read as a different
basin on a flat surface, not a wrong model.

Where the two references disagree with each **other** beyond tolerance, that is a
recorded **flag to investigate**, never a silent "pick the closer one" and never
excusable by a `glmm`-side registry entry (known exemptions: the GLMM SE-method
split and the 2-way rungs listed in the table).

## Documented divergences

`divergences.json` is the registry both consumers read — `compare.R`'s
`lme4-vs-glmm` table and the crate's own cross-engine tier
(`tests/oracle_support`). Each entry names the dataset, the quantities it covers,
the largest relative difference it covers (`max_rel`), which direction the
criterion moved, a standalone summary, and its review status.

It is built so it cannot become a blanket exemption:

| situation | outcome |
|---|---|
| over band, no entry | **FAIL** — unchanged from before the switch |
| over band, entry covers it | reported as `DOC` in its cell and again in the footer; run stays green |
| over band, past the entry's `max_rel` | **FAIL** — the divergence grew |
| entry whose dataset was compared, never fired | **FAIL** — stale entry, delete it |

Nothing about a match is silent: the cell prints `DOC`, the footer prints the
entry's direction and its review status, and the crate tier asserts the exact set
of entries that fired. The port gates below do **not** consult the registry — both
ports call the same kernel, so a miss there is a wiring bug, not a divergence.

## Layout

    manifest.json     single source of truth: the 48 curated `datasets` rungs
                      (the `m3_goldens` cells 49-52 are registered separately,
                      see below), both formula dialects, per-rung options
                      (data, link, weights_col, tier)
    run.sh            the runner -- see Running below
    compare.R         the cross-engine reference check + the two port GATES
    tol.R             every tolerance band, with its measurement history
    divergences.json  documented divergences the reference check reports rather
                      than fails on -- see Documented divergences above
    summarize_timing.R / summarize_accuracy.R / summarize_parallel.R
                      human-readable views (no gate): timing + speedups,
                      accuracy diffs, parallel-feature speedup
    summarize_memory.R
                      human-readable view (no gate) of the memory/ legs
    memory/           peak-RSS harness (no gate): memory.sh measures every
                      engine's peak resident memory over the manifest rungs
                      plus 13 large synthetic models (models.json, regenerable
                      by gen_models.py); one fit script per engine; results
                      land in results/memory/ (gitignored)
    prep/             regenerate data/ from fixed seeds: export_data.R (rungs 1-28),
                      gen_weights_data.R (rungs 29-43), gen_large_theta_data.R
                      (rungs 44-46 + the non-rung sim_binomial_zerosd fixture),
                      gen_illcond_data.R, gen_scale_data.R,
                      gen_probit_large_data.R (rung 48), gen_igauss_data.R (rungs 51-52 --
                      see the rung table's note: sim_igauss carries no manifest `rung`
                      field of its own, unlike sim_probit_large, since it is not a
                      `datasets` entry)
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
same engines, same gates, plus per-rung weight pathologies); rungs 44–45 are the
large-θ̂ tier (no `tier` field — ordinary model-shape rungs on the existing tiers,
added late so no existing rung had to be renumbered).

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
| 44 | sim_binomial_bigsd | binomial GLMM (Bernoulli, scalar RE) | large θ̂ (4.51 fitted, 3.4× the corpus's previous GLMM max) — Laplace/FD-Hessian in the hard low-information cell |
| 45 | sim_poisson_bigsd | Poisson GLMM (scalar RE) | large θ̂ (2.97 fitted) — the count-family counterpart of rung 44 |
| 46 | sim_sparse_binomial_bigsd | binomial GLMM (Bernoulli, 1 primary + 7 crossed scalar RE) | the SPARSE arm of the large-θ̂ regime — large θ̂ (3.91 fitted on `g1`) crossed with 7 crossed intercept-only extras, which is what routes it past `MAX_EXTRA_GROUPINGS` to the sparse solver. **2-way (glmm ↔ lme4) by decision, not by MixedModels limitation** — see the rung's `//` comment in `manifest.json`. Bernoulli rather than Poisson: a Poisson design in this regime pushes counts into the tens of thousands, where deviance-sum rounding noise dominates the FD Hessian's step independent of solver tuning; sparse-large-θ̂-Poisson coverage is deliberately not attempted for that reason. |
| 48 | sim_probit_large | binomial GLMM, probit link (Bernoulli, scalar RE) | the corpus's **large** probit rung — 100 groups × 96 rows = 9600 rows, p = 5, θ̂ ≈ 0.66. `cbpp_probit` (rung 22) is the only other probit fit in the suite and is 56 rows at ~3 ms, where a vectorized family kernel is pure measurement noise; this rung is what makes probit speed, and accuracy drift that needs a long row loop to show up, visible at all. The row count is capped by lme4, not by taste: above `glmerControl`'s `check.conv.nobsmax = 10000` glmer stops computing the optimizer Hessian and `engines/lme4.R`'s `vcov(use.hessian = TRUE)` hard-errors. Data from `prep/gen_probit_large_data.R`. |
| 49 | sim_cloglog_glm | binomial GLM, cloglog link | non-canonical asymmetric link, GLM arm — reuses rung 48's data |
| 50 | sim_cloglog_glmm | binomial GLMM, cloglog link (Bernoulli, scalar RE) | cloglog GLMM, dense scalar RE — the link's mixed arm |
| 51 | sim_igauss_glm | inverse-Gaussian GLM, log link | V(μ)=μ³ family kernel, log link, on the new `sim_igauss` fixture |
| 52 | sim_igauss_inv_sq_glm | inverse-Gaussian GLM, 1/μ² link | same data, non-canonical 1/μ² link — general Fisher-scoring branch |

Rungs 49–52 are `m3_goldens` cells (`stats::glm` / `lme4::glmer` only, frozen by
`engines/goldens_agq.R`), not curated 3-way rungs — they do not appear in
`compare.R`'s sweep or the status paragraphs below. This table numbers rows
sequentially for documentation, independent of any manifest `rung` field: rungs
49–52 are `m3_goldens` cells with no `rung` field of their own (sim_cloglog_glm
and sim_cloglog_glmm reuse rung 48's data; sim_igauss_glm and
sim_igauss_inv_sq_glm's underlying `sim_igauss` data has no `rung` either, since
it is not a `datasets` entry — see below). Rungs 51–52 are 2-way
(glmm ↔ `stats::glm`) by construction, not by limitation: MixedModels.jl has no
fixed-only inverse-Gaussian GLM path, and the family's mixed-model arm faults at
`assert_model_shape` (not built), so `sim_igauss` carries no `jl_formula` and no
GLMM cell — unlike cloglog, which reuses rung 48's data, `sim_igauss` is its own
fixture (`prep/gen_igauss_data.R`).

Status: rungs 44–45 got their **lme4** references on 2026-07-30
(`VALIDATION_ONLY=sim_binomial_bigsd,sim_poisson_bigsd Rscript engines/lme4.R`); rung 46 got
its **lme4** reference on 2026-08-06
(`VALIDATION_ONLY=sim_sparse_binomial_bigsd Rscript engines/lme4.R`) and is 2-way
by decision. Rung 48 got **both** references (lme4 1.1.38 and MixedModels 5.7.0) on
2026-08-23 and is 3-way; the two references agree with each other to 6e-6 on β and
1e-7 on the stddev.
Like every other rung's, those JSONs live in the gitignored `results/` tree, so a
fresh checkout still has to run `run.sh --oracles` before `compare.R` — which
iterates the lme4 result JSONs on disk — lists them. **Rung 44 got its MixedModels
leg on 2026-08-05 and is 3-way; rung 45 has none and is 2-way** (glmm ↔ lme4)
rather than the 3-way of a normal GLMM rung. Rung 45's missing leg is a package
limitation, like rung 18's and unlike the manifest decisions at 23/24: it keeps its
`jl_formula`, but MixedModels v5.7.0 throws `PosDefException` inside PIRLS on it, so
it sits in `JL_CANNOT_FIT` in `engines/mixedmodels.jl` — without that entry the
Julia engine exits non-zero and `set -e` takes the whole `--oracles` run with it.
Note `se_hessian` is a 2-way
quantity on every rung regardless — MixedModels does not compute it, so what the
missing leg costs here is the β / stddev / loglik cross-check. The green status
below is about rungs 1–43: 44–45 have not yet been through a `compare.R` sweep.

Status: all 43 referenced rungs green against both references, except the documented 2-way
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

    ./run.sh                    fit glmm (Rust) only, untimed, compare vs the EXISTING
                                lme4/MixedModels results on disk — the default gate
    ./run.sh --ports            ALSO fit the Python and R ports
    ./run.sh --oracles          ALSO refit R + Julia (oracle regeneration)
    ./run.sh --timings[=N]      ALSO time the fits (N samples, default 4; warm-up dropped,
                                median of the rest). `=N` must be attached, not spaced.
    ./run.sh --agq=K            AGQ pass: refit the manifest's `agq`-marked datasets at
                                nAGQ=K into the sibling results/<engine>_agq_* trees
    ./run.sh --prep             regenerate data/ first; implies --oracles
    ./run.sh --rust-tier2       also run the crate's oracle-tests tier first
    ./run.sh cbpp sleepstudy    restrict any of the above to named datasets

Two independent axes — **which engines fit**, and **whether fits are timed** — and
the flags compose. The whole corpus, Rust, untimed, is ~35 s.

The deviance gate described above runs inside `compare.R`, which runs at the end
of every `run.sh` invocation except an `--agq` pass — it is not tied to
`--oracles`. `measure_dev_floor.R` is a different,
rerunnable script: untimed, not part of `run.sh`, and only run by hand when
re-pinning `TOL$dev_eps`/`TOL$dev_big` in `tol.R`; it writes `results/dev_floor.csv`.

Timing is opt-in because `compare.R` reads no timing field at all: the numbers serve
only `summarize_timing.R` / `summarize_parallel.R`, which `run.sh` does not call.
All five engines implement one contract, mirrored because five languages cannot share
code: `VALIDATION_TIMINGS` unset or `0` means do not time; otherwise it **is** the
sample count (integer ≥ 2, first discarded, median of the rest). The count lives in
`run.sh` alone — no engine carries an `N_RUNS` constant any more — so `--timings=8`
reaches all five, and each result JSON still records its own `n_runs`, keeping files
timed under older counts (10, and its 100-run predecessor) self-describing. Without
`--timings` the engines write `"timing": null`, so a default run **drops any timings
already recorded in `results/`** — re-measure with `--timings` when you need them. Each timed engine also gets a `results/run_meta_<engine>.json` (machine, git
rev, `no_turbo`, core pin), which `summarize_timing.R` prints and uses to refuse to
put two machines' seconds in one comparison — seconds never transfer across boxes,
and the ratios only weakly. Real speed work belongs in `campaigns/speed-grid/`.

The ports reach the same kernel through a different binding and must match the Rust
engine to round-off, so `--ports` is what you run when a port, the lowering, or the
data changed — it cannot diverge on a kernel-only change.

`--agq=K` is the third axis, and a measurement rather than a gate: it refits only the
manifest's `agq`-marked datasets (glmm's AGQ gate — binomial/Poisson, one grouping
factor, q ≤ 3) at quadrature order K, into `results/<engine>_agq_*` so compare.R's
Laplace globs never see them, and `summarize_timing.R` prints them as the `aK` rows.
Every engine fits SERIAL, glmm included: it is the only configuration all of them can
share, since neither port builds `glmm/parallel`. MixedModels has no nAGQ knob, so `jl`
is dropped even under `--oracles`, and compare.R is skipped. Combine with `--timings`
for the numbers; the pass gets its own `results/run_meta_<engine>_agq.json`, because a
run_meta shared with the Laplace legs would claim they were fitted together.
Inner-parallelism measurement lives in `campaigns/speed-grid/agq_par_probe.rs`.

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
(swapped column, mis-sorted factor levels, dropped weights or offset), exactly
the bug class the ports' own test suites cannot see.

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

One quantity, `se_hessian_rel`, is additionally per-rung capable via `tol.R`'s
`TOL_PER_RUNG` — the corpus-wide 1e-3 is orders looser than the ≤2e-5 agreement
the crate documents, so a rung that exists to guard that agreement needs its own
number. `tol.R`'s `TOL_PER_RUNG` table owns the current overrides and their
measured sizing. Overrides may only tighten; `validate_tol_per_rung()` rejects
a widening one, an unknown rung name and an unknown quantity name at load.

## Estimator pinning

The cross-engine sweep is pinned to Laplace (lme4 `nAGQ=1`) — like-for-like
with glmm's glmer-faithful kernel. AGQ validation (`nAGQ` 7/11, scalar and
vector RE) lives in the **goldens track**: `engines/goldens_agq.R` freezes
lme4/GLMMadaptive references into `goldens/`, gated in-crate
(`src/fit/glmm_tests.rs`) at the `agq_*` bands in `tol.R`.
