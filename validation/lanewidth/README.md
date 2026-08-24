# Lane-width harness

A reusable, documented procedure for measuring how much of a `glmm` fit
result is SIMD-lane-width / summation-order dependent: run the crate's own
tests once normally, once with SIMD dispatch forced to the scalar path, and
diff the two. Built as its own deliverable — it does not depend on any
other in-flight investigation and stays useful regardless of what test
filter is pointed at it.

## Why this exists

`glmm` dispatches its SIMD transcendentals (`src/simd_transcendental.rs`)
through `pulp::Arch::new()`, which probes the CPU at runtime and picks the
widest instruction set available (AVX2/AVX-512 on x86_64, NEON on aarch64,
falling back to `Scalar` everywhere). A horizontal reduction inside those
kernels therefore sums a µarch-dependent number of partial sums, so the same
inputs on the same crate version can differ in their last few bits across
hosts — not a bug, the doc comment in `simd_transcendental.rs` names this
explicitly and the goldens are not gated on byte-identity because of it.

Most of the time that drift is unobservable. One test is not so lucky:
`fit_sparse_nb_glmm_is_pinned` (`src/sparse/tests.rs`) sits on a genuinely
ill-conditioned NB fit where a 1-ULP input perturbation already moves
`beta[0]` by ~4.8e-4 median — so lane-width alone (not a bug, not a port
error) is enough to move the pinned quantities by a comparable amount. The
original investigation found this by vendoring `pulp 0.22.2` locally,
forcing `Arch::new()` to always return `Scalar`, and comparing an AVX2
anchor machine's report against the reproduced-locally scalar drift (4.3e-4
vs 7.9e-4 on `beta[0]` — same worst element, same ordering). That was a
one-off. This directory is the reusable version: run
it against any test filter, on any host, and get a self-describing report
of what lane width was actually exercised and how much the pinned
quantities moved between a normal run and a scalar-forced run.

## What it does NOT do

It does not change the shipped `pulp` pin. `pulp` is exact-pinned at
`=0.22.2` in the crate's `Cargo.toml` — FP-affecting bumps move the
validation goldens — and the vendoring patch here is **never** applied to
that manifest. Every patched file lives inside a scratch copy the script
creates (default: `mktemp -d`, outside this repo); the real tree's
`Cargo.toml`/`Cargo.lock` are only ever read, never written.

## Files

- `run_lanewidth.sh` — the harness script (see "How to run" below).
- `pulp-0.22.2-scalar-force.patch` — a unified diff against pulp 0.22.2's
  upstream source (`src/aarch64.rs`, `src/x86.rs`, `src/wasm.rs`), applied
  by the script to a scratch-vendored copy of pulp with `patch -p1`. Each
  hunk replaces `Arch::new()`'s CPU-feature probe with an unconditional
  `Self::Scalar`, so every `pulp::WithSimd` dispatch takes the scalar path
  regardless of what the host CPU actually supports. All three platform
  files are patched so the harness works unmodified on whichever host
  µarch it's run on — only the file matching `target_arch` is actually
  compiled in, but patching all three means the same one `.patch` covers
  x86_64, aarch64 and wasm32 hosts.
- `lanewidth_probe.rs` — a small Rust source the script stages as
  `examples/lanewidth_probe.rs` into each scratch copy (never into the real
  tree). It prints `pulp::Arch::new()`'s `Debug` form, then refits the same
  sparse-NB design and data as `fit_sparse_nb_glmm_is_pinned` and prints
  `beta`, `se`, `varcorr` and `dispersion` (theta). This is both the
  "printed dispatch marker" evidence and the numeric before/after the
  investigation is actually about.
- `results/<UTC timestamp>/` — per-run output (probe stdout for both trees,
  full `cargo test` logs for both trees, and `summary.txt`). Regenerated
  every run, not committed (see `GLMM/.gitignore`), same convention as
  `validation/results/`.

## How to run

From anywhere (the script resolves its own location):

```bash
cd GLMM/validation/lanewidth
./run_lanewidth.sh                                  # default filter: the NB/sparse fit tests
./run_lanewidth.sh some_test_name another_test_name # restrict to named test(s)
SCRATCH=/path/to/dir ./run_lanewidth.sh             # pin the scratch dir (default: mktemp -d)
KEEP_SCRATCH=1 ./run_lanewidth.sh                    # don't delete the scratch dir on exit
```

Default test filter: `fit_sparse_nb_glmm_is_pinned` (the crate's most
lane-width-sensitive pin — see its doc comment in `src/sparse/tests.rs`) and
its weighted sibling `sparse_weighted_nb_matches_replicated`. Any positional
arguments
replace the default filter list; they are passed straight through as
`cargo test --lib -- <filters...>` (libtest's own multi-filter OR-match, not
cargo's single-`TESTNAME` argument), so you can pass more than one.

### What the script does, step by step

1. Copies the crate source (`GLMM/`) into two scratch trees, `normal/` and
   `scalar/`, excluding build artifacts and venvs (never source files).
2. Fetches pulp 0.22.2's source: reuses the local `~/.cargo/registry/src/`
   cache if present, otherwise runs `cargo vendor` from the `normal/`
   scratch tree (network fetch, expected and fine per the task's scope).
3. Applies `pulp-0.22.2-scalar-force.patch` to that vendored copy with
   `patch -p1`, and asserts the patch marker landed.
4. Appends a `[patch.crates-io]` table pointing `pulp` at the patched
   vendor path — **only** to `scalar/Cargo.toml`, never to `normal/` and
   never to the real tree.
5. Stages `lanewidth_probe.rs` as `examples/lanewidth_probe.rs` in both
   scratch trees and runs it in each (`cargo run --example
   lanewidth_probe`), capturing stdout to `results/<ts>/probe_{normal,
   scalar}.txt`.
6. Runs `cargo test --lib -- <filters> --nocapture` in both scratch trees,
   capturing full logs to `results/<ts>/test_{normal,scalar}.log`.
7. Writes `results/<ts>/summary.txt`: host arch/OS, which pulp-source path
   was used, the two probe lines side by side, and both `cargo test` exit
   statuses.

## How to interpret the output

**The probe is the ground-truth check that scalar forcing actually
happened.** On aarch64 hosts, `probe_normal.txt` reads
`pulp::Arch::new() = Neon(Neon { neon: Neon })` and `probe_scalar.txt`
reads `pulp::Arch::new() = Scalar`. On x86_64 the normal line instead reads
`V3(..)`/`V4(..)` depending on what the host CPU and pulp's `x86-v3`/
`x86-v4` features resolve to. **The contrast this harness measures depends
on host µarch** — "NEON vs scalar" on an Apple Silicon Mac, "AVX2/AVX-512 vs
scalar" on the x86_64 anchor machine from the original investigation. That
is why the probe line is always printed and recorded: a run is
self-describing about which lane widths it actually exercised, rather than
assuming a fixed contrast.

**The `beta`/`se`/`varcorr`/`dispersion` lines in the probe output are the
actual numeric evidence.** Diff `probe_normal.txt` against `probe_scalar.txt`
for the two scratch trees to see the movement directly, independent of
whether the pinned test's tolerance band happens to absorb it.

**A scalar-forced test failure on `fit_sparse_nb_glmm_is_pinned` is expected
evidence, not a harness bug.** That test's tolerance band (currently 3e-3,
see its doc comment) was deliberately widened to survive exactly this kind
of drift; whether a given crate revision's fixture is tight enough to fail
under lane-width forcing is a live, open question this harness exists to
let someone answer, not something it assumes either way. Read
`test_scalar.log` for the assertion's actual-vs-reference numbers when it
does fail.

**Timing is secondary evidence only.** The scalar path is not vectorized,
so the scalar-forced `cargo test` run is typically visibly slower on the
lane-width-sensitive test — useful as a sanity check that the patched pulp
was really linked in, but not a substitute for the probe or the numeric
diff.

## Worked example

Run 2026-08-02 on an aarch64-apple-darwin host (Apple Silicon Mac), scratch
dir pinned and kept for inspection:

```bash
SCRATCH=<scratch-dir> KEEP_SCRATCH=1 ./run_lanewidth.sh
```

with the default filter (`fit_sparse_nb_glmm_is_pinned
sparse_weighted_nb_matches_replicated`). The probe's dispatch marker
confirmed the contrast actually exercised on this host:

| tree   | `pulp::Arch::new()` Debug repr |
|--------|---------------------------------|
| normal | `Neon(Neon { neon: Neon })`     |
| scalar-forced | `Scalar`                 |

and the refit of `fit_sparse_nb_glmm_is_pinned`'s design moved as follows
between the two:

| quantity | normal (NEON) | scalar-forced | abs diff | rel diff |
|---|---|---|---|---|
| `beta[0]` | 0.5092910767139079 | 0.5087329325361793 | 5.58e-4 | 1.10e-3 |
| `beta[1]` | 0.4761915320842533 | 0.47620564103858526 | 1.41e-5 | 2.96e-5 |
| `se[0]` | 0.36968218481792464 | 0.36970738302146877 | 2.52e-5 | 6.8e-5 |
| `dispersion` (θ̂) | 1.3960778146152264 | 1.396160553985832 | 8.27e-5 | 5.9e-5 |
| `varcorr[0]` | 0.3819422537218528 | 0.381979557816595 | 3.7e-5 | 9.7e-5 |

`beta[0]`'s 5.58e-4 NEON-vs-scalar movement on this host is the same order
of magnitude as an earlier cross-machine investigation's figures (4.3e-4
x86_64 AVX2 host scalar-forced vs 7.9e-4 cross-machine arm64 report) —
independent confirmation that lane-width dispatch, not a
port defect, is a real contributor to that fixture's drift, this time
measured from the NEON side rather than AVX2. Both trees'
`fit_sparse_nb_glmm_is_pinned` passed (the pinned test's 3e-3 band
comfortably absorbs a 5.6e-4 movement); the failure mode this harness
would surface is a future, tighter fixture that does not.

## Reusing this for other tests or other investigations

Pass any `cargo test` filter string(s) as positional arguments. The harness
does not know anything about NB/sparse specifically past its default
filter — it is the general "does this pass/this quantity move under a
forced scalar SIMD dispatch" instrument for the whole crate, reusable for
any test filter, not a one-off tied to this investigation.
