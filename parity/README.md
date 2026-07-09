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
oracle/fit.rs      glmm harness (Cargo example `parity_fit`) -> results/glmm/
results/lme4/, results/mixedmodels/   committed reference oracles
results/glmm/      committed fits from the local glmm crate
compare.R          cross-engine agreement GATE (lme4 vs MixedModels vs glmm)
summarize_timing.R    human-readable timing view (no gate): per-fit medians + speedup vs lme4/mmjl
summarize_accuracy.R  human-readable accuracy views (no gate): diffs vs lme4 + SE-by-method (5-way)
run.sh             runs every present engine (R, jl, rust) over all datasets, then compare.R
Manifest.toml      pinned Julia packages (generated on env setup; commit for reproducibility)
```

This is a **plain part of the GLMM repo** — one repo, one `.gitignore`, no nested
sub-workspace, no isolated `Cargo.lock`. When `fit.rs` lands it is an ordinary part
of the crate's build (example / bench / small bin, decided then).

## Datasets — the roadmap rungs

| Rung | Dataset             | Family                                        | Source |
|------|---------------------|-----------------------------------------------|--------|
| 1    | Dyestuff            | gaussian (intercept)                          | lme4   |
| 2    | sleepstudy          | gaussian (intercept+slope)                    | lme4   |
| 3    | Penicillin          | gaussian (crossed)                            | lme4   |
| 4    | Pastes              | gaussian (nested)                             | lme4   |
| 5    | cbpp                | binomial GLMM                                 | lme4   |
| 6    | grouseticks         | Poisson GLMM                                  | lme4   |
| 7    | sim_slope_extra     | gaussian (crossed slopes, sparse-routed)      | sim    |
| 8    | sim_sparse_binomial | binomial GLMM (7 crossed extras, over-cap)    | sim    |
| 9    | sim_sparse_poisson  | Poisson GLMM (7 crossed extras, over-cap)     | sim    |
| 10   | Machines             | gaussian (q=3 correlated slopes)              | nlme   |
| 11   | Oats                 | gaussian (real 2-level nesting)               | nlme   |
| 12   | VerbAgg              | binomial GLMM (ungrouped 0/1, crossed)        | lme4   |
| 13   | cake                 | gaussian (`:`-interaction grouping)           | lme4   |
| 14   | Arabidopsis          | Poisson GLMM (real nested)                    | lme4   |
| 15   | sim_three_level      | gaussian (3-level nesting)                    | sim    |
| 16   | sim_max_q_slope      | gaussian (q=8, `MAX_PRIMARY_Q` boundary)      | sim    |
| 17   | sim_crossed_at_cap   | Poisson GLMM (6 crossed extras, at cap)       | sim    |
| 18   | sim_binomial_slope_crossed | binomial GLMM (2 crossed q=2 groupings) | sim    |
| 19   | sim_poisson_nested   | Poisson GLMM (3-level nesting)                | sim    |
| 20   | sim_unbalanced_nested | gaussian (3-level, heavily unbalanced)       | sim    |
| 21   | sim_nested_crossed_mix | gaussian (nested + crossed combined)        | sim    |

Rung 18 (`sim_binomial_slope_crossed`) is fit on the Rust side by the
**sparse GLMM driver** (`fit_glmm_sparse`): `classify_design` routes any
design with a slope-carrying extra grouping to `Solver::Sparse` for every
family — the dense NoZ GLMM kernel only ever implemented intercept-only
extras and never sees this shape. The aggregated binomial rows enter as
`prop = incidence/size` with prior `weights = size` (the sparse binomial
path is the one that honors prior weights). The same fit is also gated
in-crate against a frozen lme4 golden
(`fit_sparse_binomial_slope_crossed_matches_lme4` in `src/sparse.rs`,
`goldens/sim_binomial_slope_crossed.json`).

This rung is a **2-way gate only** (lme4 + glmm, no MixedModels.jl): the
pinned MixedModels.jl (v5.7.0) cannot *construct* a
`GeneralizedLinearMixedModel` for two crossed groupings that both carry a
random slope, fit with non-trivial binomial weights (`PosDefException` at
construction, before PIRLS runs — confirmed via controlled experiments to be a
package/version limitation independent of this rung's data). `compare.R`
already treats a missing `results/mixedmodels/<name>.json` as `n/a`
per-dataset, so this needed no code change either —
`results/mixedmodels/sim_binomial_slope_crossed.json` is intentionally
absent.

Rungs 8–9 are the sparse non-Gaussian GLMM's external truth (roadmap step 2):
each trips a real envelope cap (7 crossed extras > `MAX_EXTRA_GROUPINGS`) so
`classify_design` routes it to the sparse solver, and both reference engines fit
the family at Laplace — the strong two-reference condition. Their Gamma / NB
siblings (`sim_sparse_gamma` over-width q_g=5, `sim_sparse_nb` over-count) live
in the **goldens track** instead: neither oracle wires those GLMM families in
the 3-way sweep, so they follow the `sim_gamma_glmm` / `sim_nb_glmm` precedent
(lme4-only reference in `goldens/`, gated in-crate).

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
  "timing": { "fit_seconds_median": 0.0123, "n_runs": 100, "warmup_discarded": 1,
              "fits_per_sample": 1 } }
  // GLMM rungs split timing by SE method: `fit_seconds_median` is replaced by
  // `fit_seconds_median_rx` and `fit_seconds_median_hessian` (MixedModels emits only
  // _rx -- no Hessian variant). The Hessian SE (FD-Hessian / numDeriv) is the main
  // time consumer; on cbpp glmm's Hessian fit is ~1.9x its Rx fit.
```

The single `se` above is a **gaussian (LMM) rung** — its SE is profiled, one method. On
**GLMM rungs** the Laplace SE has two genuinely different variants, so `se` is replaced by
**`se_hessian`** (keeps the θ–β coupling) and **`se_rx`** (conditional on θ̂, drops it);
MixedModels computes only `se_rx`, so its GLMM files carry that field alone. This is what
lets `compare.R` measure like method against like (see Tolerances). glmm emits both (via
`WaldSe::{Hessian, Rx}`) but **no `loglik`** — its `Fit` does not expose one yet, so that
column is shown `n/a` for glmm and left ungated until it does.

Both fit scripts normalize the careful bits to one representation: **`varcomp`** to
the absolute σ scale (lme4 reports variances, MixedModels σ-relative θ); **`loglik`**
to the logLik scale (MixedModels reports the −2·logLik objective); **`reml`** is
honored from the manifest, not just recorded (MixedModels defaults to ML, which
biases varcomp downward — `null` and `sigma` absent on GLMM rungs 5–6). `beta`/`se`
follow fixed-effect design-column order (intercept first); `coef_names` asserts both
engines coded the contrasts identically.

## Timing

Each script times **only the fit call**, native high-resolution timer, in-process
loop of N (default 100), **discards the warm-up pass** (neutralizes Julia's one-time
JIT cost), reports the **median** of the rest (`fit_seconds_median`). Process startup
and CSV load are excluded. Real cross-engine numbers need the machine locked (the
**user** stabilizes + locks; this harness only records, never writes CPU sysfs) —
until then the field is indicative only. Timing does not gate correctness.

`summarize_timing.R` prints the recorded median fit time per engine (with the glmm
speedup factor); `summarize_accuracy.R` prints the accuracy diffs vs lme4 and the
SE-by-method tables.

On GLMM rungs the time is **split by SE method** (`t_rx` / `t_hess`): the Hessian SE
is the dominant cost (glmm's FD-Hessian re-solves the Laplace deviance ~O(m²) times;
lme4's `vcov(use.hessian=TRUE)` runs numDeriv), while the Rx SE is one closed-form
Schur solve. The fit underlies both, so the gap between `t_rx` and `t_hess` is the
SE-method cost — on cbpp glmm's Hessian fit is ~1.9× its Rx fit. MixedModels has only
the Rx vcov, so its GLMM time is reported under `t_rx` alone.

**Estimator is pinned to Laplace (lme4 `nAGQ=1`) in this curated sweep.** glmm's GLMM
kernel is glmer-faithful `nAGQ=1` (`src/glmm/mod.rs`), so the cross-engine sweep must be
Laplace to compare like-to-like with it. Adaptive Gauss-Hermite quadrature (`nAGQ>1`,
single scalar RE only) is more accurate but a *different* estimator — on cbpp it shifts
β ~5e-4 and the Hessian SE ~1.2% (converged by nAGQ=10). Notably lme4's AGQ Hessian SE
lands *closer* to glmm's than lme4's own Laplace SE, so that ~0.5% glmm↔lme4 Hessian gap
is partly lme4-Laplace's approximation, not a glmm defect.

AGQ comparison is **not** done here — it lives in the separate **goldens track**
(`oracle/fit_m3_goldens.R` → `goldens/`): lme4 reference fits at `nAGQ=1/7/11` for cbpp
and grouseticks, which glmm's AGQ is validated against in-crate once the M3 kernel
exposes an nAGQ knob. Keeping it out of `results/` leaves the 6-rung sweep un-expanded
(design 6) and sidesteps that AGQ is fundamentally lme4-vs-glmm (the 3-way sweep can't
require it).

## Tolerances (per-quantity, in `compare.R`)

A single threshold can't serve point estimates that agree to ~1e-4 and SEs that
legitimately differ by percent. Starting points, tuned against the first reference
run, recorded in `compare.R` — never relaxed to make an engine pass:

- **β**, **varcomp std-devs** — relative ~1e-3
- **loglik** — absolute on the shared scale, split by family: LMM ~1e-6 (the REML
  criterion is near-exact across engines, ~1e-9 observed); GLMM ~1e-3 (two optimizers
  land ~3e-6 *relative* apart on the same Laplace surface — β/varcomp confirm it is the
  same fit)
- **coef names** — compared with cosmetic formatting normalized (`period2` vs
  `period: 2`); asserts same base / levels / order, which the β gate independently confirms
- **LMM SE** (`se`) — tight ~1e-3 (all engines compute it identically)
- **GLMM SE** — split by **method**, so the comparison is like-for-like (this is what
  removed a spurious ~1–1.5% "gap" that was just lme4-Hessian measured against MM-Rx):
  - **`se_rx`** (conditional on θ̂) — all three engines compute it; gated at ~1e-3 (lme4
    vs MixedModels measured ~2e-4 on cbpp). `glmm`'s `WaldSe::Rx` joins this column.
    `compare.R` also prints **`se_rx:mm`** — the same `se_rx` checked against the *second*
    oracle (MixedModels) instead of lme4. The two references disagree ~2e-4 on their own
    Rx and `glmm` sits on the MixedModels value (`glmm`↔MM ~6e-7 on cbpp, vs lme4 ~2e-4),
    so the vs-lme4 number alone understates `glmm`'s agreement with an independent engine.
    `summarize_accuracy.R`'s per-dataset `Rx max-rel` line shows all three pairwise diffs.
  - **`se_hessian`** (keeps the θ–β coupling) — only lme4 and `glmm` compute it
    (MixedModels has none → shown `n/a`); gated at `se_hessian_rel` 1e-3, same band as
    `se_rx` (measured worst 2e-5, grouseticks). The gate sat at 3e-2 while the frozen
    references carried lme4's lagged-ldL2 `tolPwrss` artifact (~1.3% on cbpp — glmer's
    `Xwts` run one PIRLS iteration behind the mode; the 2026-07-04 hessian-curvature
    diagnosis pinned the gap as lme4's). The references are now generated at
    `tolPwrss = 1e-13` (recorded in each JSON) and `glmm`'s FD runs PIRLS at its tight
    FD-only tol, so the band reflects true engine agreement. `stddev_se_rel` (θ-block
    SE, same joint Hessian) tightened 3e-2 → 3e-3 on the same regeneration (measured
    worst 8e-4, sim_sparse_poisson — the single-step-FD vs numDeriv method floor).
    `glmm`'s default is Hessian.

## The Rust engine (incremental, by rung)

`oracle/fit.rs` (Cargo example `parity_fit`, wired via the `rust` entry in `run.sh`)
reads the same CSVs through the formula frontend (`glmm_formula::lower`) into the
stable `fit_cold` surface, emits the same schema (both `se_hessian` and `se_rx` via
`WaldSe::{Hessian, Rx}`) to `results/glmm/`, and joins the comparison against
the frozen references.

**Landed (glmm):** all nine rungs — 1 Dyestuff, 2 sleepstudy, 3 Penicillin,
4 Pastes, 5 cbpp, 6 grouseticks, 7 sim_slope_extra, 8 sim_sparse_binomial and
9 sim_sparse_poisson (the over-envelope sparse non-Gaussian GLMM rungs) — green
against both references.
