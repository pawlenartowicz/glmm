# Using `glmm` from Rust

One page, four sections. The first three add complexity in layers — cold fit,
warm fit, advanced hot loop — each building on the last. The fourth is a
shorter, self-contained note on skipping the hand-built inputs via a formula
string.

(This page covers the Rust crate only — for the Python port, see
[`TUTORIAL-PYTHON.md`](TUTORIAL-PYTHON.md).)

All inputs use **row-major** `f64` for the design matrix: `x[i * p + j]` is
row `i`, column `j`.

## 1. Cold fit — one call, one answer

`fit_cold` is the entry point for a single, one-shot fit: you hand it the
design, the response, a `ModelSpec` (structure only — family, and random-effect
topology if any), and `GroupIds` (the per-row grouping level for a mixed
model). It cold-starts the optimizer internally and returns a `Fit`.

```rust
use glmm::{fit_cold, Family, FitOptions, GroupIds, ModelSpec, ReStructure, Sizing};

// 12 rows, 2 predictors (intercept + x1), 6 groups (2 rows per group).
let n = 12;
let p = 2;
let x: Vec<f64> = (0..n).flat_map(|i| [1.0, i as f64 * 0.1]).collect();
let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.3 * (i as f64 * 0.1) + (i % 6) as f64 * 0.05).collect();

// y ~ x1 + (1 | group) — Gaussian, random intercept.
let model = ModelSpec {
    family: Family::Gaussian,
    re: Some(ReStructure {
        sizing: Sizing::FixedClusters { n_clusters: 6 },
        slopes: vec![],           // no random slope — intercept-only RE
        extra_groupings: vec![],  // one grouping factor only
    }),
};
let ids = GroupIds { primary: (0..n as u32).map(|i| i % 6).collect(), extra: vec![] };

let opts = FitOptions { target_indices: vec![0, 1], ..Default::default() }; // SE for both columns

let fit = fit_cold(&x, &y, n, p, &model, &ids, &opts);
assert!(fit.converged);
println!("beta = {:?}, se = {:?}", fit.beta, fit.se);
```

Points worth knowing at this layer:

- `family: Family::Gaussian` + `re: None` fits OLS; `re: Some(..)` fits an LMM.
  The same shape (`Family::Binomial{..}` / `Poisson{..}` / `Gamma{..}` /
  `NegativeBinomial{..}`) selects GLM (`re: None`) or GLMM (`re: Some`) —
  see the family table in [`supported_families.md`](supported_families.md).
- `ReStructure::extra_groupings` is a `Vec<Grouping>` — add one entry per
  *additional* grouping factor (crossed or nested) beyond the primary. This is
  how you fit **multiple/crossed/nested grouping factors**: give each grouping
  its own `GroupIds::extra[g]` vector aligned 1:1, declaration order.
- `FitOptions::target_indices` controls which columns get a standard error —
  SE computation has a cost, so only ask for the columns you need.
- `Fit::converged == false` signals a numerical failure (not a panic); check it
  before trusting `beta`/`se`.
- **Experimental: in-fit parallelism.** Building with the `parallel` cargo
  feature *and* setting `FitOptions { parallel_inner: true, .. }` runs the AGQ
  cluster loop and the FD-Hessian SE grid on rayon threads. Both default to
  off — the flag alone does nothing in a default build, and the feature alone
  spawns nothing until the flag is set. Results are bit-identical to serial;
  the kernels are new and their performance envelope isn't characterized yet,
  so treat this as opt-in experimentation only. Leave it off if you run your
  own parallel loop over many fits (the outer loop should own the cores).
- **The envelope.** The dense ("NoZ") solver has hard capacity limits:
  `re.slopes.len() + 1 <= 8` (primary width), up to 6 extra groupings, and
  `slopes.len() + 1 <= 4` per extra grouping (`glmm::consts::{MAX_PRIMARY_Q,
  MAX_EXTRA_GROUPINGS, MAX_EXTRA_Q}`). Within that envelope, a **Gaussian**
  extra grouping that carries a random slope is routed internally to a sparse
  solver instead (still just works — no API change). Outside the envelope, or
  a *non-Gaussian* mixed model that needs the sparse path, `fit_cold` panics
  with `unimplemented!` rather than silently misfitting — check your spec
  against the caps up front if you're building models programmatically.

## 2. Warm fit — reusing a previous solution

`fit_warm` is `fit_cold` plus one optional argument: a `StartValues` warm
start. This is the natural next step once you're fitting the *same model
shape* repeatedly with data that changes a little each time (e.g. bootstrap
resamples, an outer optimization loop, or refitting after adding a few rows) —
threading in the previous fit's `beta`/`theta` shortens the path to
convergence. `fit_cold` is exactly `fit_warm(.., None, ..)`, so nothing here
changes the *answer* — a warm start only changes how fast the optimizer gets
there; the fitted MLE is start-independent.

```rust
use glmm::{fit_warm, StartValues};

// First fit, cold.
let fit1 = fit_warm(&x, &y, n, p, &model, &ids, None, &opts);

// ...data changes slightly (new resample, new batch, refit)...
let start = StartValues { beta: fit1.beta.clone(), theta: vec![/* n_theta values from the model's RE structure */] };
let fit2 = fit_warm(&x2, &y2, n, p, &model, &ids, Some(&start), &opts);
```

Notes at this layer:

- `StartValues.beta` must have length `p`; `StartValues.theta` must match the
  model's RE θ width (primary `vech(Λ)` plus one `vech(Λ_g)` block per extra
  grouping — a mismatch panics at the call boundary, not deep inside a kernel).
- For a fixed-only model (OLS/GLM) a warm start is a no-op — there's no
  optimizer state to seed — so just call `fit_cold`/pass `None`.
- The LMM kernel only warm-starts `theta` (β is solved exactly given θ, so a
  β guess doesn't help it); GLMM warm-starts both.

## 3. Advanced — the `loop_advanced` hot-loop surface

For the tightest inner loop (thousands of near-identical fits — e.g. a Monte
Carlo power simulation), even `fit_warm`'s per-call allocation and
row-major→column-major conversion is overhead you may want to own yourself.
The `loop_advanced` cargo feature exposes the scratch-explicit kernels
directly: you allocate the workspace/scratch buffers **once**, then call the
kernel in a loop, reusing the buffers every iteration.

```toml
[dependencies]
glmm = { version = "...", features = ["loop_advanced"] }
```

```rust
use glmm::loop_advanced::{fit_lmm, LmmWorkspace};

let mut ws = LmmWorkspace::for_cluster_spec_ext(p, &model, n, &slope_cols, &extra_slope_cols);

for resample in resamples {
    ws.suff.reset();
    ws.suff.add_rows_multi(resample.x.as_ref(), &resample.y, &resample.cluster_ids, &resample.extra_ids);
    let result = fit_lmm(&mut ws, &target_indices, theta_start.as_deref());
    // ws.fit.betas / ws.theta hold this iteration's estimate; no fresh
    // allocation happened — ws is reused next iteration.
}
```

This is the **unstable** surface — no semver guarantees, and it changes in any
release. It exists to serve one class of caller (warm-start simulation loops
like MCPower's); reach for it only once you've measured that `fit_warm`'s
per-call cost actually matters for your loop. `loop_advanced` also re-exports
the OLS/GLM/GLMM/LME kernels and scratch types the same way — `fit_cold`/
`fit_warm` are a thin, allocating wrapper around exactly these.

## 4. Parsing a formula instead of building inputs by hand

Building `x`/`ModelSpec`/`GroupIds` by hand (as above) is fine for a fixed,
known model shape. If you'd rather describe the model as an R-style formula
string and a data table, the companion `glmm-formula` crate does the
lowering:

```rust
use glmm::{fit_cold, Family};
use glmm_formula::{lower, Column, Table};

let table = Table {
    columns: vec![
        ("y".into(), Column::Numeric(y_values)),
        ("x1".into(), Column::Numeric(x1_values)),
        ("group".into(), Column::Factor { labels: group_labels }),
    ],
    n,
};

let lo = lower("y ~ x1 + (1 | group)", &table, Family::Gaussian)?;
let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
```

`lower` parses the formula (`*` desugars to main effects + interaction, `A/B`
to nesting), discovers factor levels, builds the design matrix with treatment
contrasts (R's default), and returns everything `fit_cold` needs — including
`lo.col_names`, the coefficient name for each design column. Use `parse`/
`materialize` separately if you need to parse once and materialize against
several data tables (the parse step is data-free).
