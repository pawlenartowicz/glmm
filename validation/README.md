# glmm validation suite

Everything under `validation/` is dev-only and never shipped: the nested
`Cargo.toml` keeps it out of `cargo package -p glmm`.

## What is in here

| path | what it is |
|---|---|
| `grid/` | the accuracy grid: 947 model cells (816 plus the 131 ML twins of the Gaussian mixed cells), four pinned external oracles, the gates. The main reference system. |
| `data/empirical/` | committed CSVs from the datasets `lme4` and `nlme` bundle. Read byte-for-byte by every engine and by `cargo test` through `include_str!`. |
| `data/simulated/` | committed fixed-seed CSVs, regenerable byte-identically by `tools/prep/`. Also read by `include_str!`, including in CI, so the bytes are the fixture. |
| `manifest.json` | the per-rung model list for the in-crate tiers: one entry per dataset with its formula, family, link, factors and tier. Read by `tests/validation_oracle.rs`, `tools/bit_identity/dump.rs` and `tools/goldens_agq.R`. |
| `goldens/` | frozen reference results, one JSON per fixture, from lme4, R `glm` / `MASS::glm.nb`, glmmTMB (`*_tmb.json`) and GLMMadaptive. Read at run time by the `oracle-tests` tier. Regenerated only by `tools/goldens_agq.R`. The lme4 goldens record the lme4 version that wrote them in `engine_version`. 26 were written under lme4 1.1.38 and stay frozen; they are not regenerated when the lme4 pin in `grid/versions.json` moves. A re-run under lme4 2.0-6 (2026-10-27) gives the same numbers on 17 of them, moves numbers on 7 (by 1e-10 to 3e-6 relative, and `sim_gamma_glmm`'s random-effect SD by 32%), and fails on `sim_gamma_inv_glmm` and `sim_sparse_gamma`. |
| `goldens/inference/` | frozen inference references (lmerTest df and tests, the exact RLRT, ANOVA tables, predictions on new data, pairwise comparisons, effect sizes, extended-precision GLM fits on ill-conditioned designs), one subdirectory per kind (`lmertest/`, `rlrt/`, `anova/`, `predict/`, `posthoc/`, `effectsize/`, `conditioning/`), one JSON per reference. Each is registered in `manifest.json`'s `inference_references` block and written only by its generator under `tools/`: the R generators through the shared setup `tools/inference_common.R`, the `conditioning` one (`tools/goldens_glm_band.py`, mpmath) with its own provenance. |
| `results/lme4_simulated/` | the frozen `lme4` results for the weights tier, read by the `oracle-tests` tier. |
| `tools/` | tools and generators, no gates: `bit_identity/` (a "what moved" report: dump the per-rung results before and after a change and diff them; the dumps are machine-local and not committed), `lanewidth/` (SIMD lane-width sensitivity), `memory/` (peak RSS), `goldens_agq.R` (the goldens generator), `prep/` (the fixed-seed generators of `data/`, plus `gamma_agq_reference.R`, a from-scratch Gamma AGQ reference run unweighted at the console and with `--weights` to freeze `gamma_agq_reference_wts.json`), `common.rs` (the shared reader the manifest-shaped examples include). A bare run of `goldens_agq.R` or of a `prep/` script overwrites committed artifacts, so both are deliberate acts, not routine steps. |
| `tools/prep/gen_inference_data.R` | the fixed-seed generator of the inference references' shared input data: `data/empirical/Dyestuff2.csv` and the simulated designs it writes to `data/simulated/`. Run once; the CSVs are committed, and a re-run must reproduce them byte for byte. |
| `tools/prep/gen_uncorrelated_data.R` | the data of the uncorrelated and intercept-free random-effect goldens: `data/simulated/sim_uncorr_{rho,boundary}.csv` (fixed seeds), and `data/empirical/sleepstudy_d2.csv` and `Machines_dummy.csv`, which add derived columns to the empirical CSVs. Byte-identical on a re-run. |
| `data/inference/` | input data that only the inference references read, never the grid: the newdata of the `predict` entries (`tools/prep/gen_predict_data.R`, no draws) and the ill-conditioned GLM designs of the `conditioning` entries (`tools/prep/gen_glm_band_data.R`, fixed seeds). Byte-identical on a re-run. |
| `campaigns/` | two finished studies, kept rerunnable: `speed-campaign/` (optimizer cost and wall time) and `monte_carlo/` (accuracy against known truth). See `campaigns/README.md`. |

## Running the grid

From `grid/`:

```sh
./run.sh <engine> [--fast] [--timed[=N]] [--pipeline] [--ml] [--jobs N] [--label TEXT] [--keep] [cell ...]
```

- `engine` is one of `glmm`, `glmm_python`, `glmm_r`, `lme4`, `glmmtmb`,
  `glmmadaptive`, `mixedmodels`, `lmertest`. `lmertest` fits the Gaussian cells
  with a random effect with `lmerTest::lmer` and records its per-coefficient
  Satterthwaite df; its run is kept and found like an oracle's.
- `--fast` restricts the run to the cells tagged `fast` in `manifest.json` (116 of
  947).
- `--timed[=N]` times every fit: N samples, the first discarded, median of the
  rest. It implies `--fast` and `--jobs 1`, records `no_turbo`, and **refuses to
  start on an unlocked clock** unless `--timed-unlocked` is given, in which case
  the run is marked unlocked. Locking the machine is the user's job; the script
  only records the state. It also refuses to start without `taskset`, which pins
  the one worker to core 0, unless `--timed-unlocked` is given: the worker then
  runs unpinned (macOS) and the run is marked unlocked.
- `--pipeline` times the inference pipeline instead of the fit alone: the fit,
  the per-coefficient t tests and the per-term F tests, with Satterthwaite df. It
  implies `--timed` (`--timed=N` still sets N) and runs on the `fast` cells whose
  `oracles` name lmerTest, the Gaussian mixed cells. Only `glmm_r`
  (`fastglmm()`, `summary()`, `anova()`), `lmertest` (`lmerTest::lmer()`,
  `summary()`, `anova()`) and `glmmtmb` (`glmmTMB()`, `summary()`, `dof_satt()`)
  accept it; glmmTMB has no per-term F test with Satterthwaite df, so it runs
  none. Records add `term_ddf` (term → F denominator df, null for glmmTMB). The
  flag is recorded as `pipeline` in `run_meta.json`.
- `--ml` adds the cells tagged `ml` to a `--fast` or `--timed` run, which runs
  without them otherwise. They are the 131 ML twins of the Gaussian mixed cells
  (all but `sim_entangled_pair_lmm`, whose deviance gate reads a frozen REML
  value), each on its parent's data, which `glmm`, `glmm_python` and `glmm_r` fit with
  `reml = Some(false)` / `reml=False` / `REML = FALSE` (every other cell keeps the
  default estimator). A full run fits every cell, twins included, with no flag, and
  a cell run names the twins it wants, so `--ml` is refused there. The flag is
  recorded as `ml` in `run_meta.json`, and `compare.R --fast` reads it there: a
  `--fast --ml` run is gated on the fast cells and the twins. A `--fast` run made
  to check accuracy adds `--ml`.
- `--jobs N` splits the cell list into N chunks and pins worker *i* with
  `taskset` to the *i*-th core the script inherits, so `taskset -c 2,3 ./run.sh ...`
  keeps its workers on cores 2 and 3 (a timed run uses core 0). The P-cores on this
  box are 0–5, so N above 6 is refused, and so is N above the inherited core count.
- `--label TEXT` is the free-text purpose of the run; it goes into
  `run_meta.json` and into the directory name.
- `--keep` makes a finished `glmm` / `glmm_python` / `glmm_r` run the engine's kept
  run (see below). Oracle runs are always kept.
- Trailing cell names restrict the run to those cells, validated against the
  manifest first.

Then:

```sh
Rscript compare.R [--glmm-run=DIR] [--fast] [--ports-only] [--dev-floor]   # the gates; exit 0 pass, 1 fail, 2 could not run
Rscript summarize_accuracy.R [--glmm-run=DIR] [--fast]       # per-cell and per-family report
Rscript summarize_timing.R [--glmm-run=DIR] [--pipeline]     # timed runs only, provenance first
Rscript gen_manifest.R && Rscript gen.R                      # regenerate the cell list and its data
```

A fresh machine reaches the pinned oracle set with `Rscript install_oracles.R`.

## Where the references live

- **Runs** — every run is built in
  `grid/runs/<engine>/scratch/<date>_<version>_<machine>[_<label>]/`, holding
  `results.jsonl` and `run_meta.json`; a second run on the same day gets a
  counter. `scratch/` is not in git. When a kept run finishes, `grid/promote.sh`
  moves it into one of three fixed folders, the only ones git tracks:
  `runs/<engine>/last/` for a full untimed run, `runs/<engine>/last-timed/` for a
  timed run, `runs/<engine>/last-pipeline/` for a `--pipeline` run. A new timed or
  pipeline run replaces only the cells it fitted and carries the
  rest over, listing their runs under `carried` in `run_meta.json`.
  `summarize_timing.R` reports the fit-only timed runs and skips pipeline runs;
  `summarize_timing.R --pipeline` reports only the pipeline runs. The run a
  slot held before moves back to `scratch/`, and git history keeps every
  committed one: `git log -p -- grid/runs/glmm/last/` is the run history.
- **Oracle references** — an oracle's reference is `runs/<oracle>/last/`, and
  its `engine_version` must match `grid/versions.json`. A version mismatch is an
  error, not a fallback, and a `fast` or timed run is never an accuracy
  reference.
- **Pins** — `grid/versions.json`: `lme4` 2.0-6, `glmmTMB` 1.1.14,
  `GLMMadaptive` 0.9-7, `MixedModels` 5.9.0, `lmerTest` 3.2-1 (also the
  inference generators' pin), R 4.5.3, Julia 1.12.6. A bump reruns that oracle
  (or `lmertest`) into a new `last/`. The inference reference generators also
  pin `RLRsim` 3.1-9, `car` 3.1-5, `afex` 1.5-1, `numDeriv` 2016.8-1.1,
  `MASS` 7.3-65, `jsonlite` 2.0.0, `emmeans` 2.0.4, `performance` 0.18.2,
  `effectsize` 1.0.3, `insight` 1.5.4, `pbkrtest` 0.5.5 and `reformulas` 0.4.4,
  and the Python `conditioning` generator pins `mpmath` 1.4.1; bumping one of
  these regenerates no frozen reference by itself.
- **Satterthwaite df** (`compare.R` gate 5): glmm's per-coefficient df against
  `runs/lmertest/last/` on every Gaussian cell with a random effect, at
  `coef_df_rel` in `grid/tol.R`, registry-backed in gate 2's scope with the
  quantity `coef_df`. A cell where glmm's Satterthwaite pass failed (every
  `coef_df` null) is left out of the band and listed, not failed. A missing
  reference `lmertest` run (full, untimed, a record for every cell whose
  `oracles` name lmerTest) stops `compare.R` with exit 2, as a missing oracle
  run does. lmerTest is
  not one of gates 1 to 3's oracles: its fit is lme4's. The port gate also holds
  `coef_df` at `port_rel`.
- **Bands** — `grid/tol.R`, every one with the measurement and the date that set
  it. `grid/dev_align.R` holds the per-engine, per-family deviance constants, each
  a closed form with its source.
- **Documented divergences** — `grid/divergences.json`. An over-band difference
  with no entry fails; one that outgrows its entry fails; an entry that stops
  firing fails as stale. A real `glmm` defect is never registered — it stays red
  and is tracked as a bug. Most differences never reach the registry, because
  three rules decide them first: a parameter reference is an oracle at `glmm`'s
  optimum (deviance at most `dev_eps` above `glmm`'s; gate 5 reads lme4's record of
  the lmerTest fit), two such fits that both put a random-effect SD below
  `sd_boundary_abs` agree on it, and an lme4 fit whose only message is the
  singular-fit note counts as converged (`runs.R`, `reclassify_lme4_on_read`).
- **Cells with no MLE**: a cell whose manifest entry carries `no_mle` (from
  `NO_MLE` in `gen_manifest.R`, with its reason) has data that admit no
  maximum-likelihood estimate. `compare.R` passes glmm's clean refusal there,
  fails anything else (a converged fit, a panic, a timeout) and lists the cell
  in its own `no-MLE` block. It is a property of the data, never a way to
  excuse a fit glmm got wrong.
- **AGQ cells with two or more random effects per group**: glmm and GLMMadaptive
  place the quadrature grid differently there, so their deviances are two
  different approximations. Gate 1 therefore evaluates GLMMadaptive's rule at
  glmm's fitted point (`ga_rule_dev` in `grid/dev_align.R`) and compares that
  with GLMMadaptive's own deviance; the cell's line says so. glmm's own deviance
  on these cells also moves with how the design is standardized, so a glmm run
  compared cell by cell with an earlier glmm run reads them against
  `dev_eps_agq_vec` in `grid/tol.R`, not `dev_eps`.
- **In-crate references** — `goldens/` and `results/lme4_simulated/`, frozen, read
  by `cargo test --features oracle-tests`.
- **Inference references**: `goldens/inference/<kind>/<name>.json`, frozen, each
  registered in `manifest.json`'s `inference_references` (an entry carrying
  `pending_reference` is registered but not yet frozen). Only the generators
  write them (`VALIDATION_ONLY=<name>` writes one entry; a bare run rewrites
  every reference of its kind). `tests/inference_references.rs` checks the
  block against the tree. Their bands are `TOL_INFERENCE` in `grid/tol.R`, and
  an over-band end-to-end difference passes only under an `oracle-tier` entry of
  `grid/divergences.json` whose `cell` is the reference's name.

## What runs where

| Check | Reads | Runs in CI? |
|---|---|---|
| `cargo test` (six feature configs) | `goldens/`, `data/empirical/`, `data/simulated/` via `include_str!` | **yes** — every push |
| `cargo test --test grid_reference` | `grid/manifest.json`, `grid/data/fast/`, `data/`, `grid/runs/glmm/last/` | **yes** — part of the `cargo test` matrix |
| `port-gate` (Rust vs Python vs R on the `fast` subset) | `grid/manifest.json`, `grid/data/fast/`, `data/`, `grid/compare.R --ports-only` | **yes** — its own job; installs `jsonlite`, no statistics package |
| Anova port gate (the per-term tables and model comparisons of a fixed call list through Rust, Python and R) | `tools/anova_ports/` (`anova_ports.rs`, `anova_ports.py`, `anova_ports.R`, `compare_anova_ports.R`) | **yes**: a step of the port-gate job, run even when the grid port gate fails |
| `cargo test --test inference_references` (the inference references' registration) | `manifest.json`, `goldens/inference/`, `data/`, `tools/` | **yes**: every push, part of the `cargo test` matrix |
| `cargo test --features oracle-tests` (the cross-engine tier) | `goldens/`, `results/lme4_simulated/`, `data/`, `grid/divergences.json` | no — local only |
| `tests/inference_oracle.rs` and the in-crate inference tests under `oracle-tests` (the inference end-to-end tier) | `goldens/inference/`, `manifest.json`, `data/`, `grid/divergences.json` | no: local only; whether it gets a CI job is decided again from the tier's measured runtime on the release tree |
| `cargo test --features montecarlo-tests` (the seeded Monte Carlo studies: the level of the exact RLRT, and `calibration_gates`, the calibration of glmm's tests and intervals against the truth on 216 simulated designs; see [`../documentation/validation.md`](../documentation/validation.md#calibration-against-the-truth)) | its own seeded draws | no: local only; about 50 minutes on eight cores (`calibration_gates` alone took 48 on 2026-10-14), nearly all of it the calibration study |
| `grid/run.sh <oracle>` (an oracle reference run) | everything | no — local only; needs R, Julia and the pinned oracle versions |
| `grid/run.sh … --timed` | the `fast` subset | no — local only, and only on a clock-locked machine |
| `grid/compare.R` (the five gates) | the committed oracle runs plus a chosen glmm run | no — local only |
| campaigns | own manifests and results | no — finished studies, rerun by hand |
| `tools/memory/memory.sh`, `tools/lanewidth/` | `data/`, their own models | no — local only, measurement not gate |

Two rules hold that table together. **CI never runs an external statistics package**: the
two checks that do run there compare the crate against numbers this repo already holds — the
committed glmm run and the two ports. **An oracle comparison is local**, because it needs
lme4, glmmTMB, GLMMadaptive, MixedModels.jl and lmerTest at their pinned versions, and because its
verdict is a finding to read rather than a build to break.
