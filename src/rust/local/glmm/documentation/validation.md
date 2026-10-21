# How glmm is validated

## The accuracy grid

`glmm` is validated against a grid of 923 model cells in `validation/grid/`. Each
cell fixes a family, a link, a random-effect structure, a size, a balance and a
regime, and — where the data is simulated — records the parameters it was
generated from. A cell is fitted by `glmm` and by every external engine that can
fit it, on the same data and the same formula, and the results land in one JSONL
record per cell per engine.

Four external engines are the oracles, each pinned to an exact version:

| Oracle | Version | Cells it fits | Estimator |
|---|---|---|---|
| R `lme4` | 2.0-6 | every family; negative binomial through `glmer.nb` (911 of 923 cells) | Laplace, scalar AGQ |
| R `glmmTMB` | 1.1.14 | every family but inverse-Gaussian, no AGQ cells (892 of 923 cells) | Laplace |
| Julia `MixedModels.jl` | 5.9.0 | gaussian, binomial logit/probit, Poisson (587 of 923 cells) | Laplace |
| R `GLMMadaptive` | 0.9-7 | non-gaussian, one grouping factor (28 of 923 cells, all AGQ) | AGQ |

A fifth engine, R `lmerTest` 3.2-1, fits the 259 Gaussian mixed-model cells
(the 130 REML cells and their ML twins) for one quantity only, the
per-coefficient Satterthwaite degrees of freedom. Its fit is lme4's, so the
deviance and parameter checks below never read it.

129 of the cells are ML twins (tag `ml`): each refits one of the REML Gaussian
mixed-model cells by maximum likelihood (`reml = false`) on that cell's data,
and lme4, glmmTMB, MixedModels.jl and lmerTest fit it with their own ML switch.
Every Gaussian mixed-model cell has a twin except `sim_entangled_pair_lmm`,
whose deviance check reads a frozen REML value. A full run fits the twins;
`run.sh --ml` adds them to a `--fast` or `--timed` run.

The pins are in `validation/grid/versions.json`. A version bump is a deliberate
event: that oracle is rerun over the whole grid into a **new** run directory, and
no existing run is ever edited.

## No engine is "the" reference

`glmm` is not gated against one chosen engine. `validation/grid/compare.R` applies
five checks:

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
5. **Satterthwaite df against lmerTest.** On every Gaussian mixed-model cell,
   `glmm`'s per-coefficient df are compared with lmerTest's. Out of band passes
   only under a registered entry, as in check 2.

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
`cargo test --features oracle-tests`. A third tier checks glmm's tests and
intervals against the generating truth (below).

## Running it yourself

From `validation/grid/`:

```sh
./run.sh glmm --fast              # fit the fast subset with glmm
./run.sh lme4                     # fit the whole grid with an oracle
Rscript compare.R                 # the five gates over the newest runs
Rscript summarize_accuracy.R      # per-cell and per-family report, no gate
```

Every run gets its own directory under `validation/grid/runs/<engine>/scratch/`,
which git ignores. An oracle run, and a `glmm` run started with `--keep`, then
replaces the engine's kept run in `last/` (full runs) or `last-timed/` (timed
runs). `--timed` needs a locked CPU clock and refuses to start without one.
See [`../validation/README.md`](../validation/README.md) for the directory layout
and the flags. The JSONL record's fields are defined in
`validation/grid/engines/common.rs`'s `base_record`.

## Calibration against the truth

The grid shows that glmm fits what other engines fit; it does not show that a
p-value or an interval means what it claims. A Monte Carlo tier does that on
simulated data, with the generating truth as the only reference: under a true null
hypothesis a test at level α should reject in a fraction α of the datasets, and an
interval at level 0.95 should cover the true value in 95% of them. The tier measures
both for the tests and intervals glmm gives on GLMMs and Gaussian mixed models,
gates them where theory says the method is calibrated, and reports the direction and
size of the error everywhere else. What the measurement means for each test, and the
advice that follows, is in
[`hypothesis-tests.md`](hypothesis-tests.md#how-well-calibrated-are-these-tests).

### The design

The GLMM designs follow the simulation design of Li, X. & Signorelli, M. (2026), *A
Comparison of R Packages for Estimating Generalized Linear Mixed Models*,
arXiv:2606.15933v1, in our own code and with our own seeds (no script, data or
result of that study is used). Subjects are observed at 1 to `M` distinct times
drawn from `0, 1/(M − 1), …, 1`; each subject has a binary `group` (probability
0.4); the linear predictor is
`β0 + 0.3·time − 0.2·group + 0.1·time·group + b0 + b1·time`, with a random-intercept
standard deviation of 0.8 and, on the random-slope designs, a slope standard
deviation of 0.4 and a correlation of 0.2. Four axes are crossed, 216 designs
(cells):

| Axis | Levels |
|---|---|
| clusters `K` | 10, 20, 30, 50, 100, 200 |
| most visits per subject `M` | 3, 9, 17 (mean cluster size 2, 5, 9) |
| response | binary with prevalence about 0.54, 0.22, 0.10 (`β0` = 0.1, −1.5, −2.5); counts with mean about 1.65, 0.34 (`β0` = 0.1, −1.5); Gaussian with `σ = 1` |
| random part | `(1 \| id)` or `(1 + time \| id)` |

A cell is named `k<K>_m<M>_<response>_<random>`, the response one of `bern54`,
`bern22`, `bern10`, `pois165`, `pois034`, `gauss` and the random part `rdi`
(`(1 | id)`) or `rdis` (`(1 + time | id)`): `k50_m3_bern10_rdi` has 50 clusters of
mean size 2, binary data with prevalence 0.10 and a random intercept. Each cell has
2000 datasets. One dataset carries several responses on the same design and random
effects, so that each null hypothesis is true on one of them: no `time:group`
effect; no `group` or `time:group` effect; a three-level subject factor `f3` with no
effect (a two-df null); on the `(1 | id)` cells, no random-slope variance; and, on a
response drawn without random effects, no random-intercept variance. The generator
is deterministic and platform independent: a cell and a dataset index give the same
data on every machine.

Measured per cell, at α = 0.01, 0.05 and 0.10 and at the levels 0.90, 0.95 and 0.99:
the Wald z rows and the Type II and Type III Wald χ² rows, the likelihood-ratio
tests of `compare()` (one-df terms, the two-df factor, and the two-df test of
`group` and `time:group` together), the Satterthwaite t and F rows and the joint F,
the ML likelihood-ratio test of a Gaussian fixed part, the 50:50 mixture tests of a
variance component, the Wald intervals (GLMM) and the Satterthwaite intervals
(Gaussian); and power, bias and RMSE. Three arms are reported only: the `time:group`
tests and the Wald intervals of a GLM fitted to a response drawn without random
effects on the same design, the `time:group` tests of the GLMM fitted with 11
adaptive quadrature nodes on the `(1 | id)` cells, and the coverage of
profile-likelihood intervals, which glmm does not compute. Every quantity comes
through the code a user reaches: the fits, `compare()`, the per-term tables and the
construction `confint()` uses.

A dataset whose fixed design is aliased (`group` constant, or one group observed at
a single time) is left out and counted: at most 3.05% of a cell's datasets
(`k10_m3_bern10_rdi`), none at `K ≥ 30`. A fit that does not converge leaves its
quantities out; on sparse binary data that is mostly separation, where no
maximum-likelihood estimate exists. Rates are over the datasets kept.

### What is gated

A rate `q̂` over `R_kept` datasets passes against its nominal value `q0` when
`|q̂ − q0| ≤ δ·q0 + 4.5·√(q0(1 − q0)/R_kept)`; a coverage is read on its miss rate.
`δ` is 0 for the Satterthwaite rows, which claim calibration in small samples, and
0.25 (a relative error of 25%, plus noise) for the methods that are calibrated only
as the number of clusters grows. A gated quantity also fails when more than 1% of
its datasets lose a fit.

| Method | Gated on | Side |
|---|---|---|
| Satterthwaite t, F rows and joint F, Satterthwaite intervals | Gaussian cells, `K ≥ 20` | two-sided, `δ = 0` |
| the same | Gaussian cells, `K = 10` | not anti-conservative, `δ = 0` |
| GLMM Wald (z rows, Type II χ², intervals) and likelihood-ratio tests, one-df terms and the two-df test | `K ≥ 50`, except the 12 binary cells with `M = 3` and prevalence 0.22 or 0.10 (78 cells) | two-sided, `δ = 0.25` |
| ML likelihood-ratio test of a Gaussian fixed part | Gaussian cells, `K ≥ 30` | two-sided, `δ = 0.25` |
| 50:50 mixture tests of a variance component | the cells with enough information per cluster: Gaussian; counts except mean 0.34 at `M = 3`; binary at `M ≥ 9` except prevalence 0.10 at `M = 9`; less the cells whose fits failed too often in a pilot | not anti-conservative, `δ = 0.25`; on the Gaussian random-slope test at `K ≥ 50` also not conservative at α = 0.10, `δ = 0` |

The cell lists are `mc_data::bands` in `src/inference/mc_data.rs`. Everything else is
reported only: the GLMM rows of the two-df factor `f3`, the GLM rows, the AGQ arm,
the profile intervals, power, bias and RMSE, and every cell outside the table. The
set was fixed from theory and a pilot (on other seeds) before the measurement and
was not re-selected on its result. It holds 4416 checks, with about 0.014 failures
expected from Monte Carlo noise, and the measurement passes all of them.

**Moved to reported after the measurement.** Two parts of the set as first fixed
missed, for reasons that are not glmm bugs, and were moved to reported after the
measurement; no band was widened.

- The Satterthwaite F of `f3`, a three-level factor that varies between clusters, on
  every random-slope Gaussian cell with `K ≤ 30` (9 cells). It is anti-conservative
  there: 0.098 at α = 0.05 on `k10_m3_gauss_rdis`, 0.073 on `k20_m17_gauss_rdis`,
  0.078 on `k30_m9_gauss_rdis`. lmerTest's Satterthwaite F gives the same rates on
  the same datasets (0.101 and 0.078 on the first and the last), decision for
  decision where both engines reach the same REML optimum. Kenward-Roger is
  conservative at `K = 10` (0.027 on `k10_m3_gauss_rdis`) and slightly above nominal
  at `K = 30` (0.0615 on `k30_m9_gauss_rdis`). The row missed on 5 of the 9 cells;
  the whole regime was moved. On the same cells the one-df `group` row and the joint
  F pass.
- The random-intercept test of `k20_m9_bern22_rdi`: its fits failed on 21 of the
  2000 datasets (1.05%, over the 1% cap), every one separated; its rate passes
  (0.022 at α = 0.05).

**Fits with no more observations than random effects.** Under `(1 + time | id)` with
mean cluster size 2 (`M = 3`), 52% to 58% of a cell's datasets have no more
observations than random effects. lme4 refuses such a fit by default
(`check.nobs.vs.nRE`); glmm fits it and says nothing. The model is weakly identified
there, not unidentified (a subject seen at three distinct times identifies the
residual variance). No gated result depends on these fits: the `f3` excess above
holds on the datasets of `k10_m3_gauss_rdis` with more observations than random
effects too (0.085 at α = 0.05).

### What it found

The Satterthwaite rows and intervals pass their gates (two-sided from `K = 20`, not
anti-conservative at `K = 10`), apart from the `f3` regime above, and the ML
likelihood-ratio test of a Gaussian fixed part passes from `K = 30` (at `K = 10` it
rejects 0.079 on `k10_m9_gauss_rdis` to 0.116 on `k10_m3_gauss_rdis` at α = 0.05).
The 50:50 mixture tests pass their one-sided gate everywhere it applies, and on GLMM
fits they are mostly conservative: at α = 0.05 on the gated cells the random-slope
test rejects 0.010 (`k200_m9_bern54_rdi`) to 0.055 (`k200_m17_bern10_rdi`), median
0.022, and the random-intercept test 0.006 (`k10_m17_pois165_rdi`) to 0.050
(`k200_m9_pois034_rdi`), median 0.028. The Gaussian random-slope test rejects 0.030
(`k10_m17_gauss_rdi`) to 0.066 (`k10_m3_gauss_rdi`).

GLMM Wald and likelihood-ratio tests at few clusters, pooled over the 18 binary and
the 12 count cells at each `K`: the rejection rate at α = 0.05, both tests decided
on the same datasets, over the four one-df null terms (`time:group` twice, `group`
in Type III and in Type II) and, apart, the two-df factor `f3`:

| `K` | binary, Wald | binary, LR | count, Wald | count, LR | `f3` binary, Wald / LR | `f3` count, Wald / LR |
|---|---|---|---|---|---|---|
| 10 | 0.0302 | 0.0551 | 0.0549 | 0.0582 | 0.0741 / 0.0934 | 0.1907 / 0.1493 |
| 20 | 0.0380 | 0.0559 | 0.0526 | 0.0554 | 0.0736 / 0.0853 | 0.1151 / 0.0958 |
| 30 | 0.0398 | 0.0542 | 0.0529 | 0.0544 | 0.0649 / 0.0773 | 0.0878 / 0.0793 |
| 50 | 0.0410 | 0.0512 | 0.0507 | 0.0512 | 0.0593 / 0.0698 | 0.0731 / 0.0692 |
| 100 | 0.0439 | 0.0503 | 0.0541 | 0.0540 | 0.0535 / 0.0596 | 0.0599 / 0.0574 |
| 200 | 0.0472 | 0.0505 | 0.0511 | 0.0510 | 0.0515 / 0.0546 | 0.0545 / 0.0532 |

On binary data Wald is conservative and the likelihood-ratio test anti-conservative
at few clusters; at α = 0.01 the pooled rates at `K = 10` are 0.0040 and 0.0118. On
counts the two agree on the one-df terms, and Wald is the more anti-conservative on
the two-df factor. The pooled rates hide a wide spread between cells: at `K = 10`
the `time:group` Wald test rejects 0.000 on `k10_m3_bern10_rdis` and 0.052 on
`k10_m17_pois165_rdis`. On binary data with about two observations per cluster and
prevalence 0.10 the Laplace fit fails at every `K`, which the mixture tests and the
intercept's interval show most (details and advice in
[`hypothesis-tests.md`](hypothesis-tests.md#how-well-calibrated-are-these-tests)).

### Running the tier

The tier is in-crate, behind the feature `montecarlo-tests`:

```sh
cargo test --lib --features montecarlo-tests calibration_gates -- --nocapture
MC_TIER_OUT=cells.jsonl cargo test --lib --features montecarlo-tests \
    calibration_measurement -- --ignored --nocapture
```

`calibration_gates` measures every cell at 2000 datasets and asserts the gated set;
a miss names the cell, the quantity, the rate, the band and its distance in standard
errors. `calibration_measurement` measures without gating, at `MC_R` datasets per
cell (default 2000). `cargo test --features montecarlo-tests` runs
`calibration_gates` together with the level study of the exact restricted
likelihood-ratio test. Measuring the 216 cells (`calibration_measurement`) took 43
minutes with eight threads; `calibration_gates` runs the same measurement and takes
about as long; the GLMM cells are nearly all of it.

- `MC_THREADS`: the number of threads (default: the available parallelism). The
  datasets are split across threads and the counts summed, so the result does not
  depend on it.
- `MC_CELLS`: a comma-separated list of cell names; only those are measured (and
  gated).
- `MC_R`: the datasets per cell of `calibration_measurement`; `calibration_gates`
  refuses any value but 2000, the one its bands are built for.
- `MC_TIER_OUT`: a file to which each cell's counts are written as one JSON line.

The run prints one `CELL` line per cell, then one `TEST` line per test (`R_kept`,
aliased and failed datasets, the rates at α = 0.01, 0.05, 0.10 and their distance
from nominal in standard errors) and one `COVER` line per interval, so
`grep '^TEST k50_m3_bern10_rdi '` gives one cell's rejection rates. The bands are
`mc_data::bands` in `src/inference/mc_data.rs`, which also holds the grid and the
generator.
