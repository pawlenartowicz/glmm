# How glmm is validated

## The accuracy grid

`glmm` is validated against a grid of 773 model cells in `validation/grid/`. Each
cell fixes a family, a link, a random-effect structure, a size, a balance and a
regime, and — where the data is simulated — records the parameters it was
generated from. A cell is fitted by `glmm` and by every external engine that can
fit it, on the same data and the same formula, and the results land in one JSONL
record per cell per engine.

Four external engines are the oracles, each pinned to an exact version:

| Oracle | Version | Cells it fits | Estimator |
|---|---|---|---|
| R `lme4` | 2.0-6 | every family; negative binomial through `glmer.nb` | Laplace, scalar AGQ |
| R `glmmTMB` | 1.1.14 | every family but inverse-Gaussian (771 of 773 cells) | Laplace |
| Julia `MixedModels.jl` | 5.9.0 | gaussian, binomial logit/probit, Poisson | Laplace |
| R `GLMMadaptive` | 0.9-7 | non-gaussian, one grouping factor | AGQ |

The pins are in `validation/grid/versions.json`. A version bump is a deliberate
event: that oracle is rerun over the whole grid into a **new** run directory, and
no existing run is ever edited.

## No engine is "the" reference

`glmm` is not gated against one chosen engine. `validation/grid/compare.R` applies
four checks:

1. **Deviance against the best oracle (hard).** On a cell's estimator arm every
   engine minimises the same objective, so after convention alignment a converged
   deviance worse than the best oracle's is a real regression and fails. An
   equal-or-better deviance always passes. A gap larger than the mismatch
   threshold in either direction fails as a suspected convention bug, not a fit
   result.
2. **Parameters against the nearest oracle.** Fixed effects, standard errors by
   matching method, random-effect standard deviations and correlations pass when
   they are within band of at least one oracle that fitted the cell. Out of band
   against all of them passes only under an entry in
   `validation/grid/divergences.json`: an undocumented difference fails, one that
   outgrows its entry fails, and an entry that stops firing fails as stale.
3. **Error against the truth, per family (hard).** On the generated cells the
   per-cell error against the generating parameters is averaged per family and
   estimator arm, for `glmm` and for every oracle. `glmm` fails the family if it
   is worse than the best oracle by more than the paired band. This is the check
   that makes an oracle's own defect visible rather than contagious.
4. **Port gates (hard).** The Python and R packages wrap the same Rust kernel, so
   they are compared against the Rust engine at a round-off band. A miss is a
   wiring bug, never a divergence.

Where two oracles disagree with **each other**, that is recorded as a flag to
investigate. It is never resolved by picking whichever one sits closer to `glmm`.

## What is covered

Gaussian, binomial (logit, probit, cloglog; Bernoulli and trials>1), Poisson,
Gamma (log and inverse) and negative binomial, across intercept-only, correlated
slopes, nested, crossed and sparse-routed structures, from 60 to 30000 rows, plus
offsets, prior weights, AGQ cells, boundary cells where the true random-effect
standard deviation is zero, and the `lme4`/`nlme` example datasets. Inverse-Gaussian
has no mixed-model path in the kernel and appears as GLM cells only.

Two smaller tiers sit alongside the grid and need no external engine: tight pins on
numbers proven correct before they were recorded, asserted on every push, and a
cross-engine tier over the frozen single-engine references in
`validation/goldens/` and `validation/results/lme4_simulated/`, run with
`cargo test --features oracle-tests`.

## Running it yourself

From `validation/grid/`:

```sh
./run.sh glmm --fast              # fit the fast subset with glmm
./run.sh lme4                     # fit the whole grid with an oracle
Rscript compare.R                 # the four gates over the newest runs
Rscript summarize_accuracy.R      # per-cell and per-family report, no gate
```

Every run gets its own never-overwritten directory under
`validation/grid/runs/<engine>/`. A `glmm` run lands in `scratch/` unless started
with `--keep`. `--timed` needs a locked CPU clock and refuses to start without one.
See [`../validation/README.md`](../validation/README.md) for the directory layout
and the flags. The JSONL record's fields are defined in
`validation/grid/engines/common.rs`'s `base_record`.
