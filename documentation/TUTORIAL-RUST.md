# Using `glmm` from Rust

One page, four sections. The first three add complexity in layers — cold fit,
warm fit, advanced hot loop — each building on the last. The fourth is a
shorter, self-contained note on skipping the hand-built inputs via a formula
string.

(This page covers the Rust crate only — for the Python port, see
[`TUTORIAL-PYTHON.md`](TUTORIAL-PYTHON.md); for the R port,
[`TUTORIAL-R.md`](TUTORIAL-R.md).)

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
- **The envelope.** The dense ("NoZ") solver has capacity limits:
  `re.slopes.len() + 1 <= 8` (primary width), up to 6 extra groupings, and
  `slopes.len() + 1 <= 4` per extra grouping (`glmm::consts::{MAX_PRIMARY_Q,
  MAX_EXTRA_GROUPINGS, MAX_EXTRA_Q}`). Within that envelope, a **Gaussian**
  extra grouping that carries a random slope is routed internally to a sparse
  solver instead (still just works — no API change). Outside the envelope, a
  mixed model with any extra grouping carrying a random slope, or a design
  with too many crossed levels, is likewise routed to the sparse solver
  automatically — every family fits either way, with no reachable panic.

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
Carlo power simulation), even `fit_warm`'s per-call allocation is overhead you
may want to own yourself. The `loop_advanced` cargo feature lets you split the
fit in two: `build_workspace` classifies the design and allocates the solver's
per-shape buffers **once**, then `fit_on` solves each draw on that workspace.
(Not everything is hoisted: the offset vectors are copied per call, and the
sparse and negative-binomial routes still allocate their own buffers per draw —
they get the shared routing, not the reuse win.)

```toml
[dependencies]
glmm = { version = "...", features = ["loop_advanced"] }
```

```rust
use glmm::loop_advanced::{build_workspace, fit_on};

// The spec you pass to `build_workspace` must carry the REAL random-effect
// level counts, not the placeholders `fit_cold` accepts — the stable entry
// derives them from the ids for you, the loop tier does not.
let mut ws = build_workspace(&sized_model, n_max, p, &opts);

for resample in resamples {
    let view = fit_on(&mut ws, &resample.x, &resample.y, &resample.ids, start.as_ref(), &opts);
    // Hot-loop reads, straight off the workspace slots — no `Fit` allocated:
    let significant = view.t_sq()[0] > crit && view.converged();
    // Or pay for the full result when you actually need it:
    // let fit = view.into_fit(&resample.x, &resample.y, n, p, &model, &opts);
}
```

`fit_on` panics rather than misroutes: the design shape (rows, predictors, RE
level counts) and the options frozen at build (`nagq`, whether weights and
offsets are present, `parallel_inner`) must match the workspace every call. Row
counts may vary below the `n_max` you built at.

What it does **not** do is validate your data. `fit_warm` checks start-value
lengths, weight and offset lengths and finiteness, and it drops rank-deficient
fixed-effect columns and refits. `fit_on` does none of that — it trusts the
caller. A short `StartValues.theta` leaves the trailing entries at the *previous*
fit's θ̂, so the same call can answer differently depending on how old the
workspace is. A weights vector longer than `n` is accepted and skews the LMM
deviance and log-likelihood; non-finite weights or offsets turn the fit into
NaN, and the GLMM route panics on a length mismatch instead. Check these once
outside your loop.

This is the **unstable** surface — no semver guarantees, and it changes in any
release. It exists to serve one class of caller (warm-start simulation loops
like MCPower's); reach for it only once you've measured that `fit_warm`'s
per-call cost actually matters for your loop. `fit_cold`/`fit_warm` route
through this same `build_workspace`/`fit_on` core — they allocate a throwaway
workspace per call and always assemble the full `Fit`. That shared core is the
point: the solver cannot drift between the stable entry and the loop tier. The
answers still can, on one input — an aliased fixed-effect design. `fit_warm`
drops the aliased columns and returns a reduced fit; `fit_on` runs the full
design and returns NaN with `converged: false`.
`loop_advanced` also re-exports the individual OLS/GLM/GLMM/LME kernels and
their scratch types, a level below this.

## 4. Parsing a formula instead of building inputs by hand

Building `x`/`ModelSpec`/`GroupIds` by hand (as above) is fine for a fixed,
known model shape. If you'd rather describe the model as an R-style formula
string and a data table, the `glmm::formula` module (the `formula` feature, on
by default) does the lowering:

```rust
use glmm::formula::{lower, Column, Table};
use glmm::{fit_cold, Family};

let table = Table {
    columns: vec![
        ("y".into(), Column::Numeric(y_values)),
        ("x1".into(), Column::Numeric(x1_values)),
        ("group".into(), Column::factor_from_labels(&group_labels)),
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

If you only need the kernel — the parse-once/fit-many hot path — take
`glmm = { version = "0.1", default-features = false }`; the formula module
disappears and the crate links no `regex`.
