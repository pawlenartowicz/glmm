# boundary-probe: what changes a fit after a variance is pinned?

A fit that ends with a variance component pinned at zero can be worked on
after the pin. This campaign measured, on corpora of boundary fits, the
candidate mechanisms: an outward probe of each pinned variance with a restart
of the search when the deviance drops (on both fit paths), a post-pin pass on
the LMM path (an undo check, a stationarity probe and a polish), and on the
GLMM path the re-pin of a trough polish's endpoint. For each it counted how
many fits move, by how much the deviance changes, the extra evaluations per
pinned fit, and which `converged`, `singular` and `pinned` flags flip. Of
these, the tree keeps:

- the pin inside the canonical fold, on every route (LMM and GLMM);
- on the GLMM path, the probe of the free coordinates at the kept pin of the
  search's endpoint, and one curvature-scaled trough polish whose endpoint is
  pinned again;
- the polish's budget exit: a polish that runs out of its evaluation budget
  reports `converged = false`.

The measurement compares two builds of the crate; the crate itself gets no
switch. Both arms are the speed campaign's `grid_fit` driver
(`../speed-campaign/fit.rs`) run over the same generated corpora:

- **before**: a tree without the mechanism under test.
- **after**: a tree with it. Run on the current tree, this arm measures the
  combination the tree keeps (the list above), not any one mechanism alone,
  and its records carry the `post_pin_searches` counter field.

Both arms build with the `counters` feature, so the only difference between
them is the crate source. Fits are deterministic, so neither arm needs a
locked machine; `GRID_CELL_BUDGET=0` fits each cell once and no time is read.

## Corpora

Seeded generators in `gen/`, each byte-identical on re-run. Every dataset is
written as `<case_id>.csv` (or a shared CSV named by the cell's `data` field)
plus one grid manifest per corpus; `gen/gen_common.R` holds the shared
helpers. Each generator's header gives its full design.

| Generator | Corpus | Fits | Design |
|---|---|---|---|
| `gen_lmm.R` | `lmm` | 7,200 | Gaussian; one grouping with two slopes, crossed intercepts (dense), slope plus crossed intercept, slope on the extra grouping (sparse); G 10, 30, 100; every SD 0.5, one SD 0, two SDs 0; REML and ML |
| `gen_q3.R` | `q3` | 4,500 | `y ~ x1 + x2 + (1 + x1 + x2 \| g1)`; Gaussian, Bernoulli, Poisson; G 20, 50, 200; the same three truth patterns |
| `gen_r1.R` | `r1` | 12,600 | blocks of at most two terms, `(1 \| g)`, `(1 + x \| g)`, `(1 \| g) + (1 \| h)`; Bernoulli, Poisson; G 20, 50, 200; varied SD 0, 0.05, 0.2 |
| `gen_r1.R` | `r1_agq5` | 8,406 | the single-grouping `r1` datasets at `nagq = 5` |
| `gen_precision.R` | `precision` | 2,400 | Poisson log-means e² to e¹⁶, binomial with 10 to 1e5 trials; one SD at 0 or 9e-5; crossed, nested, random-slope shapes |

`validation/tools/prep/gen_boundary_pin_data.R` builds the in-crate
boundary-pin test dataset `sim_poisson_precise_two_zero_slopes.csv` with these
generators' helpers, a precise Poisson design outside the corpora.

Data and results are not committed: the generators write to `$S/data/`, the
arms to `$S/<arm>/`, with `$S` any scratch directory.

## Commands

From the crate root, R with `jsonlite` on `R_LIBS`:

```bash
for g in gen_lmm.R gen_q3.R gen_r1.R gen_precision.R; do
  Rscript validation/campaigns/boundary-probe/gen/$g $S/data
done
python3 -c 'import json, sys; json.dump({"cells": [c for f in sys.argv[2:] for c in json.load(open(f))["cells"]]}, open(sys.argv[1], "w"))' \
  $S/manifest.json $S/data/manifest_{lmm,q3,r1,r1_agq5,precision}.json

# one arm (before or after), and a second run of it for determinism
GRID_MANIFEST=$S/manifest.json GRID_DATA_DIR=$S/data GRID_OUT=$S/before/records.jsonl \
  GRID_CELL_BUDGET=0 cargo run --release -p validation --example grid_fit --features counters

python3 validation/campaigns/boundary-probe/compare.py --same $S/before/records.jsonl $S/before2/records.jsonl
python3 validation/campaigns/boundary-probe/compare.py $S/before/records.jsonl $S/after/records.jsonl > $S/compare.md
```

`grid_fit` appends and skips cells already in its output, so an interrupted
arm resumes by running the same command again; a fresh arm needs an empty
output file. The whole corpus (35,106 fits) runs in minutes in one process.

`compare.py --same` passes when two runs wrote the same records in the same
order apart from the timing fields. `compare.py BEFORE AFTER` refuses unless
both arms hold the same `case_id`s, and reports per corpus and route: fits,
failures, fits pinned before, fits moved, the deviance change, fits worse by
more than 64·eps·max(1, |dev|), extra evaluations per pinned fit, flag flips,
and, from the after arm's counter, the `post_pin_searches` histogram.

## The accuracy grid's singular cells

The change can move only fits that pin, so each arm also runs the grid's
singular cells of the kept run on the same machine, from `validation/grid/`:

```bash
CELLS=$(python3 -c 'import json; print(" ".join(sorted({r["cell"] for r in map(json.loads, open("runs/glmm/last/results.jsonl")) if r.get("singular")})))')
GRID_MACHINE=<machine> ./run.sh glmm --label "boundary probe before" $CELLS
```

Both arms use the same `GRID_MACHINE` name and the default watchdog (600 s): a
shorter one times out on the slowest singular cells, and a timeout record
compares nothing. A named-cell run is not gated by `compare.R`; its records are
compared with the other arm's run on the same machine, cell by cell.
