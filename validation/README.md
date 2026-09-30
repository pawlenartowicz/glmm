# glmm validation suite

Everything under `validation/` is dev-only and never shipped: the nested
`Cargo.toml` keeps it out of `cargo package -p glmm`.

## What is in here

| path | what it is |
|---|---|
| `grid/` | the accuracy grid: 773 model cells, four pinned external oracles, the gates. The main reference system. |
| `data/empirical/` | committed CSVs from the datasets `lme4` and `nlme` bundle. Read byte-for-byte by every engine and by `cargo test` through `include_str!`. |
| `data/simulated/` | committed fixed-seed CSVs, regenerable byte-identically by `tools/prep/`. Also read by `include_str!`, including in CI, so the bytes are the fixture. |
| `manifest.json` | the per-rung model list for the in-crate tiers: one entry per dataset with its formula, family, link, factors and tier. Read by `tests/validation_oracle.rs`, `tools/bit_identity/dump.rs` and `tools/goldens_agq.R`. |
| `goldens/` | frozen single-engine reference results for the shapes one engine covers (AGQ tiers, Gamma/NB families, probit). Read at run time by the `oracle-tests` tier. Regenerated only by `tools/goldens_agq.R`. |
| `results/lme4_simulated/` | the frozen `lme4` results for the weights tier, read by the `oracle-tests` tier. |
| `tools/` | tools and generators, no gates: `bit_identity/` (the byte-identity dump), `lanewidth/` (SIMD lane-width sensitivity), `memory/` (peak RSS), `goldens_agq.R` (the goldens generator), `prep/` (the fixed-seed generators of `data/`, plus `gamma_agq_reference.R`, a from-scratch Gamma AGQ reference run unweighted at the console and with `--weights` to freeze `gamma_agq_reference_wts.json`), `common.rs` (the shared reader the manifest-shaped examples include). A bare run of `goldens_agq.R` or of a `prep/` script overwrites committed artifacts, so both are deliberate acts, not routine steps. |
| `campaigns/` | two finished studies, kept rerunnable: `speed-campaign/` (optimizer cost and wall time) and `monte_carlo/` (accuracy against known truth). See `campaigns/README.md`. |

## Running the grid

From `grid/`:

```sh
./run.sh <engine> [--fast] [--timed[=N]] [--jobs N] [--label TEXT] [--keep] [cell ...]
```

- `engine` is one of `glmm`, `glmm_python`, `glmm_r`, `lme4`, `glmmtmb`,
  `glmmadaptive`, `mixedmodels`.
- `--fast` restricts the run to the cells tagged `fast` in `manifest.json` (58 of
  773).
- `--timed[=N]` times every fit: N samples, the first discarded, median of the
  rest. It implies `--fast` and `--jobs 1`, records `no_turbo`, and **refuses to
  start on an unlocked clock** unless `--timed-unlocked` is given, in which case
  the run is marked unlocked. Locking the machine is the user's job; the script
  only records the state.
- `--jobs N` splits the cell list into N chunks and pins worker *i* with
  `taskset -c i`. The P-cores on this box are 0–5, so N above 6 is refused.
- `--label TEXT` is the free-text purpose of the run; it goes into
  `run_meta.json` and into the directory name.
- `--keep` puts a `glmm` / `glmm_python` / `glmm_r` run under `runs/<engine>/`
  instead of `runs/<engine>/scratch/`. Oracle runs are always kept.
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

- **Oracle runs** — `grid/runs/<oracle>/<date>_<version>_<machine>[_<label>]/`,
  holding `results.jsonl` and `run_meta.json`. A run never overwrites; a second
  run on the same day gets a counter. The reference for an oracle is its newest
  committed run whose `run_meta.json` says `subset: full` and whose
  `engine_version` matches `grid/versions.json`. A version mismatch is an error,
  not a fallback, and a `fast` or timed run is never an accuracy reference.
- **Pins** — `grid/versions.json`: `lme4` 2.0-6, `glmmTMB` 1.1.14,
  `GLMMadaptive` 0.9-7, `MixedModels` 5.9.0, R 4.5.3, Julia 1.12.6. A bump reruns
  that oracle into a new directory and edits no old one.
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
  with GLMMadaptive's own deviance; the cell's line says so.
- **In-crate references** — `goldens/` and `results/lme4_simulated/`, frozen, read
  by `cargo test --features oracle-tests`.

## What runs where

| Check | Reads | Runs in CI? |
|---|---|---|
| `cargo test` (six feature configs) | `goldens/`, `data/empirical/`, `data/simulated/` via `include_str!` | **yes** — every push |
| `cargo test --test grid_reference` | `grid/manifest.json`, `grid/data/fast/`, `data/`, the newest committed run under `grid/runs/glmm/` | **yes** — part of the `cargo test` matrix |
| `port-gate` (Rust vs Python vs R on the `fast` subset) | `grid/manifest.json`, `grid/data/fast/`, `data/`, `grid/compare.R --ports-only` | **yes** — its own job; installs `jsonlite`, no statistics package |
| `cargo test --features oracle-tests` (the cross-engine tier) | `goldens/`, `results/lme4_simulated/`, `data/`, `grid/divergences.json` | no — local only |
| `grid/run.sh <oracle>` (an oracle reference run) | everything | no — local only; needs R, Julia and the pinned oracle versions |
| `grid/run.sh … --timed` | the `fast` subset | no — local only, and only on a clock-locked machine |
| `grid/compare.R` (the four gates) | the committed oracle runs plus a chosen glmm run | no — local only |
| campaigns | own manifests and results | no — finished studies, rerun by hand |
| `tools/memory/memory.sh`, `tools/lanewidth/` | `data/`, their own models | no — local only, measurement not gate |

Two rules hold that table together. **CI never runs an external statistics package**: the
two checks that do run there compare the crate against numbers this repo already holds — the
committed glmm run and the two ports. **An oracle comparison is local**, because it needs
lme4, glmmTMB, GLMMadaptive and MixedModels.jl at their pinned versions, and because its
verdict is a finding to read rather than a build to break.
