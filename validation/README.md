# glmm validation suite

Everything under `validation/` is dev-only and never shipped: the nested
`Cargo.toml` keeps it out of `cargo package -p glmm`.

## What is in here

| path | what it is |
|---|---|
| `grid/` | the accuracy grid: 794 model cells, four pinned external oracles, the gates. The main reference system. |
| `data/empirical/` | committed CSVs from the datasets `lme4` and `nlme` bundle. Read byte-for-byte by every engine and by `cargo test` through `include_str!`. |
| `data/simulated/` | committed fixed-seed CSVs, regenerable byte-identically by `tools/prep/`. Also read by `include_str!`, including in CI, so the bytes are the fixture. |
| `manifest.json` | the per-rung model list for the in-crate tiers: one entry per dataset with its formula, family, link, factors and tier. Read by `tests/validation_oracle.rs`, `tools/bit_identity/dump.rs` and `tools/goldens_agq.R`. |
| `goldens/` | frozen reference results, one JSON per fixture, from lme4, R `glm` / `MASS::glm.nb`, glmmTMB (`*_tmb.json`) and GLMMadaptive. Read at run time by the `oracle-tests` tier. Regenerated only by `tools/goldens_agq.R`. |
| `goldens/inference/` | frozen inference references (lmerTest df and tests, the exact RLRT, ANOVA tables), one subdirectory per kind (`lmertest/`, `rlrt/`, `anova/`), one JSON per reference. Each is registered in `manifest.json`'s `inference_references` block and written only by its step's generator under `tools/`, through the shared setup `tools/inference_common.R`. |
| `results/lme4_simulated/` | the frozen `lme4` results for the weights tier, read by the `oracle-tests` tier. |
| `tools/` | tools and generators, no gates: `bit_identity/` (a "what moved" report: dump the per-rung results before and after a change and diff them; the dumps are machine-local and not committed), `lanewidth/` (SIMD lane-width sensitivity), `memory/` (peak RSS), `goldens_agq.R` (the goldens generator), `prep/` (the fixed-seed generators of `data/`, plus `gamma_agq_reference.R`, a from-scratch Gamma AGQ reference run unweighted at the console and with `--weights` to freeze `gamma_agq_reference_wts.json`), `common.rs` (the shared reader the manifest-shaped examples include). A bare run of `goldens_agq.R` or of a `prep/` script overwrites committed artifacts, so both are deliberate acts, not routine steps. |
| `tools/prep/gen_inference_data.R` | the fixed-seed generator of the inference references' shared input data: `data/empirical/Dyestuff2.csv` and the simulated designs it writes to `data/simulated/`. Run once; the CSVs are committed, and a re-run must reproduce them byte for byte. |
| `campaigns/` | two finished studies, kept rerunnable: `speed-campaign/` (optimizer cost and wall time) and `monte_carlo/` (accuracy against known truth). See `campaigns/README.md`. |

## Running the grid

From `grid/`:

```sh
./run.sh <engine> [--fast] [--timed[=N]] [--jobs N] [--label TEXT] [--keep] [cell ...]
```

- `engine` is one of `glmm`, `glmm_python`, `glmm_r`, `lme4`, `glmmtmb`,
  `glmmadaptive`, `mixedmodels`, `lmertest`. `lmertest` fits the Gaussian cells
  with a random effect with `lmerTest::lmer` and records its per-coefficient
  Satterthwaite df; its run is kept and found like an oracle's.
- `--fast` restricts the run to the cells tagged `fast` in `manifest.json` (116 of
  794).
- `--timed[=N]` times every fit: N samples, the first discarded, median of the
  rest. It implies `--fast` and `--jobs 1`, records `no_turbo`, and **refuses to
  start on an unlocked clock** unless `--timed-unlocked` is given, in which case
  the run is marked unlocked. Locking the machine is the user's job; the script
  only records the state. It also refuses to start without `taskset`, which pins
  the one worker to core 0, unless `--timed-unlocked` is given: the worker then
  runs unpinned (macOS) and the run is marked unlocked.
- `--jobs N` splits the cell list into N chunks and pins worker *i* with
  `taskset -c i`. The P-cores on this box are 0–5, so N above 6 is refused.
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
Rscript summarize_timing.R [--glmm-run=DIR]                  # timed runs only, provenance first
Rscript gen_manifest.R && Rscript gen.R                      # regenerate the cell list and its data
```

A fresh machine reaches the pinned oracle set with `Rscript install_oracles.R`.

## Where the references live

- **Runs** — every run is built in
  `grid/runs/<engine>/scratch/<date>_<version>_<machine>[_<label>]/`, holding
  `results.jsonl` and `run_meta.json`; a second run on the same day gets a
  counter. `scratch/` is not in git. When a kept run finishes, `grid/promote.sh`
  moves it into one of two fixed folders, the only ones git tracks:
  `runs/<engine>/last/` for a full untimed run, `runs/<engine>/last-timed/` for a
  timed run. A new timed run replaces only the cells it fitted and carries the
  rest over, listing their runs under `carried` in `run_meta.json`. The run a
  slot held before moves back to `scratch/`, and git history keeps every
  committed one: `git log -p -- grid/runs/glmm/last/` is the run history.
- **Oracle references** — an oracle's reference is `runs/<oracle>/last/`, and
  its `engine_version` must match `grid/versions.json`. A version mismatch is an
  error, not a fallback, and a `fast` or timed run is never an accuracy
  reference.
- **Pins** — `grid/versions.json`: `lme4` 2.0-6, `glmmTMB` 1.1.14,
  `GLMMadaptive` 0.9-7, `MixedModels` 5.9.0, `lmerTest` 3.2-1 (also the
  inference generators' pin), R 4.5.3, Julia 1.12.6. A bump reruns that oracle
  (or `lmertest`) into a new `last/`.
- **Satterthwaite df** (`compare.R` gate 5): glmm's per-coefficient df against
  `runs/lmertest/last/` on every Gaussian cell with a random effect, at
  `coef_df_rel` in `grid/tol.R`, registry-backed in gate 2's scope with the
  quantity `coef_df`. A cell where glmm's Satterthwaite pass failed (every
  `coef_df` null) is left out of the band and listed, not failed. While
  `coef_df_rel` is NA, gate 5 prints `NOT RUN (pending: <reason>)` (with or
  without a `lmertest` run), every other gate runs, and the exit status is
  theirs. lmerTest is
  not one of gates 1 to 3's oracles: its fit is lme4's. The port gate also holds
  `coef_df` at `port_rel`.
- **Bands** — `grid/tol.R`, every one with the measurement and the date that set
  it. `grid/dev_align.R` holds the per-engine, per-family deviance constants, each
  a closed form with its source.
- **Documented divergences** — `grid/divergences.json`. An over-band difference
  with no entry fails; one that outgrows its entry fails; an entry that stops
  firing fails as stale. A real `glmm` defect is never registered — it stays red
  and is tracked as a bug.
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
| `cargo test --test inference_references` (the inference references' registration) | `manifest.json`, `goldens/inference/`, `data/`, `tools/` | **yes**: every push, part of the `cargo test` matrix |
| `cargo test --features oracle-tests` (the cross-engine tier) | `goldens/`, `results/lme4_simulated/`, `data/`, `grid/divergences.json` | no — local only |
| `tests/inference_oracle.rs` and the in-crate inference tests under `oracle-tests` (the inference end-to-end tier) | `goldens/inference/`, `manifest.json`, `data/`, `grid/divergences.json` | no: local only; whether it gets a CI job is decided again from the tier's measured runtime on the release tree |
| `cargo test --features montecarlo-tests` (seeded Monte Carlo level studies) | its own seeded draws | no: local only; whether it gets a CI leg is decided from the measured runtime of its first study (the level of the exact RLRT), and this row changes with that decision |
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
