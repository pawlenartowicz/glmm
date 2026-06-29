# GLMM parity evaluator

Fits the **same statistical model on the same data** with **R `lme4`**, **Julia
`MixedModels.jl`**, and (later) the Rust `glmm` crate, then compares **estimates**
(correctness) and **fit-time** (speed). Built **R + Julia first, Rust last**: the
reference truth lives where the datasets live. Two independent reference engines
agreeing within tolerance is the strong-truth condition `glmm` is later held to.

## The oracle is sacred

> The committed `data/*.csv` and the committed R / Julia result JSONs **are** the
> frozen oracle. When `glmm` is later compared against them, on any disagreement
> **`glmm` is presumed wrong** — investigate and fix `glmm`. A reference result is
> regenerated **only** if the reference *model spec* itself is proven wrong (wrong
> formula, family, link), with a recorded justification. **Never** relax a tolerance
> or edit a reference to make `glmm` pass.

Committing the data and the reference JSON *is* the freeze — no machinery needed. A
second guard rides along: `glmm` must match **both** engines; where `lme4` and
`MixedModels.jl` themselves disagree beyond tolerance, that is a recorded **flag to
investigate**, never a silent "pick the closer one." The one known, exempt
disagreement is the GLMM standard-error method split (see Tolerances below).

## Layout

```
manifest.json      single source of truth: curated datasets + per-dataset model (both dialects)
prep/export_data.R pull lme4-origin datasets -> data/*.csv (run once; output committed)
data/*.csv         neutral input, committed -- EVERY engine reads these bytes
oracle/fit.R       read csv + manifest, fit lme4, time fit-only, write results/lme4/<ds>.json
oracle/fit.jl      same with MixedModels.jl -> results/mixedmodels/<ds>.json
oracle/fit.rs      (later) glmm harness -- extends, does not restructure
results/<engine>/  one JSON per (dataset x engine)
compare.R          lme4-vs-MixedModels agreement check now; + glmm later
run.sh             runs every present engine over all datasets; adding "rust" = one list entry
Manifest.toml      pinned Julia packages (generated on env setup; commit for reproducibility)
```

This is a **plain part of the GLMM repo** — one repo, one `.gitignore`, no nested
sub-workspace, no isolated `Cargo.lock`. When `fit.rs` lands it is an ordinary part
of the crate's build (example / bench / small bin, decided then).

## Datasets — the roadmap rungs

| Rung | Dataset     | Family                       | Source |
|------|-------------|------------------------------|--------|
| 1    | Dyestuff    | gaussian (intercept)         | lme4   |
| 2    | sleepstudy  | gaussian (intercept+slope)   | lme4   |
| 3    | Penicillin  | gaussian (crossed)           | lme4   |
| 4    | Pastes      | gaussian (nested)            | lme4   |
| 5    | cbpp        | binomial GLMM                | lme4   |
| 6    | grouseticks | Poisson GLMM                 | lme4   |

Each dataset is exported **once** from its canonical source (lme4) to neutral CSV;
Julia and Rust read that CSV, **not** their own ecosystem's built-in copy — this is
what guarantees byte-identical input and sidesteps row-order / factor-coding / NA
differences. The manifest's `factors` list names the columns every engine re-coerces
to categorical (R `factor` / Julia `String` + `DummyCoding`); both use the first
sorted level as the contrast base, asserted in `compare.R` via `coef_names`.

InstEval and salamander are **deferred** — added when their rungs need them.

## Running

```sh
./run.sh            # fit all present engines + compare
./run.sh --prep     # regenerate committed data/*.csv first, then fit + compare
```

**R** needs `lme4` + `jsonlite`. **Julia** runs in this dir's pinned env; set it up
once (this writes `Project.toml` + `Manifest.toml` — commit them for a reproducibly
regenerable oracle):

```sh
julia --project=. -e 'using Pkg; Pkg.add(["MixedModels","CSV","DataFrames","JSON3"]); Pkg.instantiate()'
```

## Result schema

One file per (dataset × engine), uniform field names so comparison is mechanical:

```jsonc
{ "dataset": "sleepstudy", "engine": "lme4", "engine_version": "1.1-35",
  "family": "gaussian", "reml": true, "rung": 2, "converged": true, "singular": false,
  "coef_names": ["(Intercept)", "Days"],
  "estimates": {
    "beta": [...], "se": [...], "sigma": 25.59, "loglik": -871.81,
    "varcomp": [ { "group": "Subject", "terms": ["(Intercept)", "Days"],
                  "stddev": [24.74, 5.92], "corr": [[1, 0.07], [0.07, 1]] } ] },
  "timing": { "fit_seconds_min": 0.0123, "n_runs": 10, "warmup_discarded": 1 } }
```

Both fit scripts normalize the careful bits to one representation: **`varcomp`** to
the absolute σ scale (lme4 reports variances, MixedModels σ-relative θ); **`loglik`**
to the logLik scale (MixedModels reports the −2·logLik objective); **`reml`** is
honored from the manifest, not just recorded (MixedModels defaults to ML, which
biases varcomp downward — `null` and `sigma` absent on GLMM rungs 5–6). `beta`/`se`
follow fixed-effect design-column order (intercept first); `coef_names` asserts both
engines coded the contrasts identically.

## Timing

Each script times **only the fit call**, native high-resolution timer, in-process
loop of N (default 10), **discards the warm-up pass** (neutralizes Julia's one-time
JIT cost), reports the **min** of the rest. Process startup and CSV load are excluded.
Real cross-engine numbers need the machine locked (the **user** stabilizes + locks;
this harness only records, never writes CPU sysfs). Timing does not gate correctness.

## Tolerances (per-quantity, in `compare.R`)

A single threshold can't serve point estimates that agree to ~1e-4 and SEs that
legitimately differ by percent. Starting points, tuned against the first reference
run, recorded in `compare.R` — never relaxed to make an engine pass:

- **β**, **varcomp std-devs** — relative ~1e-3
- **loglik** — absolute on the shared logLik scale
- **LMM SE** — tight ~1e-3 (all engines compute it identically)
- **GLMM SE** — **exempt** from the lme4-vs-MixedModels gate: lme4 keeps the θ–β
  coupling, MixedModels.jl drops it (~3% smaller) — a documented method difference,
  recorded not flagged. `glmm`'s default keeps the coupling, so its GLMM SE **is**
  gated — against **lme4 only**.

## Adding the Rust engine (deferred)

`oracle/fit.rs` + one `ENGINES` entry (`rust`) in `run.sh`; it reads the same CSVs,
emits the same schema to `results/glmm/`, and joins the existing comparison against
the frozen references. Rungs 5–6 are GLMM, reachable in `glmm` only via the unstable
`mcpower` feature until the roadmap wires GLMM into stable `fit`.
