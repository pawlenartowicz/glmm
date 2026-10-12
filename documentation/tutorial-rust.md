# Using `glmm` from Rust

One page, four sections. The first three add complexity in layers — cold fit,
warm fit, advanced hot loop — each building on the last. The fourth is a
shorter, self-contained note on skipping the hand-built inputs via a formula
string.

(This page covers the Rust crate only — for the Python port, see
[`tutorial-python.md`](tutorial-python.md); for the R port,
[`tutorial-r.md`](tutorial-r.md).)

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
// y = 1 + 0.3·x1, plus a shift per group and a term that varies within a group
// (so the model does not fit y exactly).
let y: Vec<f64> = (0..n)
    .map(|i| 1.0 + 0.3 * (i as f64 * 0.1) + (i % 6) as f64 * 0.05 + ((i * 7) % 5) as f64 * 0.02)
    .collect();

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

let mut opts = FitOptions::default();
opts.target_indices = vec![0, 1]; // SE for both columns

let fit = fit_cold(&x, &y, n, p, &model, &ids, &opts);
assert!(fit.converged());
println!("beta = {:?}, se = {:?}", fit.beta, fit.se);
```

Points worth knowing at this layer:

- `family: Family::Gaussian` + `re: None` fits OLS; `re: Some(..)` fits an LMM,
  by REML unless `FitOptions::reml` is `Some(false)` (ML).
  The same shape (`Family::Binomial{..}` / `Poisson{..}` / `Gamma{..}` /
  `NegativeBinomial{..}`) selects GLM (`re: None`) or GLMM (`re: Some`) —
  see the family table in [`supported_families.md`](supported_families.md).
- `ReStructure::extra_groupings` is a `Vec<Grouping>` — add one entry per
  *additional* grouping factor (crossed or nested) beyond the primary. This is
  how you fit **multiple/crossed/nested grouping factors**: give each grouping
  its own `GroupIds::extra[g]` vector aligned 1:1, declaration order.
- `FitOptions::target_indices` controls which columns get a standard error —
  SE computation has a cost, so only ask for the columns you need.
- `fit.converged() == false` signals a numerical failure (not a panic); check
  it before trusting `beta`/`se`. It reads `fit.diagnostics.converged` —
  everything the fit reports about *itself* (convergence, singularity, the
  aliased-column mask, the θ boundary state, any solver notes, and how the
  gradient search ended: `n_grad`, `outer_stop`, `outer_decrement`) lives on
  `fit.diagnostics`, with `converged()`/`singular()`/`aliased()` kept as
  forwarding accessors for the three most-read ones.
- **Experimental: in-fit parallelism.** Building with the `parallel` cargo
  feature *and* setting `opts.parallel_inner = true` runs the AGQ
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

**Degrees of freedom and contrasts.** A Gaussian mixed fit also carries
lmerTest's Satterthwaite degrees of freedom: one derivative pass after the
fit, with no refit, on by default (`FitOptions::satterthwaite`). They give t
tests where `se` alone gives only z:

```rust
use glmm::inference::contrast::{ContrastError, NoDfReason};

// Satterthwaite df of each target coefficient: the `df` column of lmerTest's
// `summary()`.
println!("df = {:?}", fit.coef_df);

// t test of one contrast lᵀβ = 0, `l` over all p columns: here the slope.
let t = fit.contrast_t(&[0.0, 1.0]).expect("a Gaussian LMM fit carries df");
println!("t = {} on {} df, p = {}", t.t, t.df, t.p_value);

// F test of Lβ = 0, `L` row-major (rows × p): here both coefficients.
let f = fit.contrast_f(&[1.0, 0.0, 0.0, 1.0], 2).expect("df");
println!("F({}, {}) = {}, p = {}", f.num_df, f.den_df, f.f, f.p_value);

// With the pass switched off, the contrasts say why there is no df.
let mut off = opts.clone();
off.satterthwaite = false;
let fit_off = fit_cold(&x, &y, n, p, &model, &ids, &off);
assert!(fit_off.coef_df.iter().all(|d| d.is_nan()));
assert_eq!(
    fit_off.contrast_t(&[0.0, 1.0]),
    Err(ContrastError::NoDf(NoDfReason::Disabled))
);
```

- `coef_df` is NaN on columns outside `target_indices`, on aliased columns,
  and on every family without a df (binomial, Poisson, negative binomial,
  every GLMM). OLS and the Gamma and inverse-Gaussian GLMs with φ estimated
  carry the residual df `n − p` instead, whatever `satterthwaite` says, and
  their contrasts are the classical t and F.
- `contrast_t` returns lmerTest's `contest1D` (`estimate`, `se`, `t`, `df`,
  `p_value`), `contrast_f` its `contestMD` (`f`, `num_df`, `den_df`,
  `p_value`). Both read only what the pass left on the `Fit`. A contrast that
  touches an aliased column gives NaN fields and a
  `ContrastNote::AliasedColumn` in `notes`; a fit with no df gives
  `ContrastError::NoDf` with the reason, never a silent z test.
- When the pass cannot take the curvature at face value, the fit carries
  `Note::SatterthwaiteCurvature` in `diagnostics.notes`
  ([`warnings.md`](warnings.md) has the cases). The method is in
  [`algorithms-lmm.md`](algorithms-lmm.md#satterthwaite-degrees-of-freedom).

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
let start = StartValues { beta: fit1.beta.clone(), theta: vec![/* n_theta values from the model's RE structure */], dispersion: None };
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
(Not everything is hoisted: the GLMM offset vector is copied per call, and the
fixed-effect negative-binomial route allocates its own buffers per draw.)

```toml
[dependencies]
glmm = { version = "...", features = ["loop_advanced"] }
```

```rust
use glmm::loop_advanced::{build_workspace, fit_on, spec_sized_from_ids_pub};

// The spec you pass to `build_workspace` must carry the REAL random-effect
// level counts, not the placeholders `fit_cold` accepts. `spec_sized_from_ids_pub`
// derives them from the ids, as the stable entry does. It may also reorder the
// groupings: `fit_on` takes the ids it returns, and `perm` records the order.
let (sized, sized_ids, perm) = spec_sized_from_ids_pub(&model, &ids);
let mut ws = build_workspace(&sized, perm, n_max, p, &opts);

for draw in draws {
    let view = fit_on(&mut ws, &draw.x, &draw.y, &sized_ids, start.as_ref(), &opts);
    // Hot-loop reads, straight off the workspace slots — no `Fit` allocated:
    let significant = view.t_sq()[0] > crit && view.converged();
    // Or pay for the full result when you actually need it:
    // let fit = view.into_fit(&draw.x, &draw.y, &sized_ids, n, p, &model, &opts);
}
```

The loop tier runs no Satterthwaite pass by itself: a `Fit` that
`FitView::into_fit` assembles from an LMM workspace carries no df, and its
contrasts return `NoDfReason::StateMissing`. Ask for the df on the draws that
need them, after `fit_on` (or `fit_on_hinted`) and with the view dropped:

```rust
use glmm::loop_advanced::satterthwaite;

let mut df_out = vec![0.0; p];
let view = fit_on(&mut ws, &draw.x, &draw.y, &sized_ids, start.as_ref(), &opts);
let converged = view.converged();
drop(view);
if converged {
    // Gaussian LMM workspaces only; NaN off the target columns.
    satterthwaite(&mut ws, &mut df_out)?;
}
```

It gives the stable path's `coef_df` for the draw, reads θ̂ without writing
it (so it and `curvature_hint` answer the same in either order), and keeps
its buffers on the workspace: a call after the first allocates only the
pseudo-inverse's blocks. Any other workspace returns
`Err(ContrastError::NoDf(NoDfReason::NotGaussianMixed))`, and a failed pass
returns `NoDfReason::Failed` with `df_out` all NaN. On a rank-deficient draw
the df are as untrustworthy as β̂, so read the diagnostics first, as below.

`fit_on` panics rather than misroutes: the design shape (rows, predictors, RE
level counts) and the options frozen at build (`nagq`, the criterion `reml`
resolves to on an LMM, whether weights and offsets are present,
`parallel_inner`) must match the workspace every call. Row
counts may vary below the `n_max` you built at.

What it does **not** do is validate your data. `fit_warm` checks start-value
lengths, weight and offset lengths and finiteness, and it drops rank-deficient
fixed-effect columns and refits. `fit_on` does none of that — it trusts the
caller. A `StartValues.beta` or `StartValues.theta` of the wrong width is
dropped without an error, and that part cold-starts. A weights vector longer
than `n` is accepted and skews the LMM deviance and log-likelihood; non-finite
weights or offsets turn the fit into NaN, and the GLMM route panics on a length mismatch instead. Check these once
outside your loop.

This is the **unstable** surface — no semver guarantees, and it changes in any
release. It exists to serve one class of caller (warm-start simulation loops
like MCPower's); reach for it only once you've measured that `fit_warm`'s
per-call cost actually matters for your loop. `fit_cold`/`fit_warm` route
through this same `build_workspace`/`fit_on` core — they allocate a throwaway
workspace per call and always assemble the full `Fit`. That shared core is the
point: the solver cannot drift between the stable entry and the loop tier. The
answers still can, on one input — a rank-deficient fixed-effect design.
`fit_warm` runs the alias gate first, drops the redundant column and returns a
reduced fit. `fit_on` skips that gate on purpose — that is part of what you are
buying — and hands the full design to the solver, so one of two things comes
back: NaN with `converged: false`, or real numbers with an enormous standard
error and `ill_conditioned` set. Which one is settled by the arithmetic on that
draw, and you cannot predict it from the design or the route: the same route,
given the same kind of duplicated column at two sample sizes, can land on
opposite sides.

That makes checking your job. Call `FitView::diagnostics()` on **every** draw and
read `converged` and `ill_conditioned` before you use the numbers — a returned
number is not evidence of a good fit, and `converged` alone misses every draw
that fit through badly conditioned. `pivot_col` names the column to look at when
the flag fires.
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
`glmm = { version = "0.5", default-features = false }`; the formula module
disappears and the crate links no `regex`.
