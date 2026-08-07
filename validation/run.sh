#!/usr/bin/env bash
# Run the fit engines over the manifest datasets, then the agreement check.
#
# TWO INDEPENDENT AXES, not a ladder. WHICH ENGINES fit, and WHETHER fits are timed.
# They compose freely; the defaults are the cheap corner of both.
#
#   engines   glmm (Rust) always. `--ports` adds the Python and R ports. `--oracles`
#             adds R/lme4 and Julia/MixedModels, which also REFITS the references
#             every other engine is compared against.
#   timing    off unless `--timings`. compare.R reads no timing field whatsoever --
#             the numbers exist only for summarize_timing.R / summarize_parallel.R,
#             which this script does not call. Timing the default run therefore paid
#             ~20 minutes for a ~35-second question, which is why it is now opt-in.
#
#   ./run.sh                 glmm (Rust) only, untimed, compared against the EXISTING
#                             results/lme4_*/mixedmodels_* JSONs on disk. The default
#                             gate: one fit per rung per SE method, whole corpus.
#   ./run.sh --ports         ALSO fit the Python and R ports. They reach the same
#                             kernel through a different binding (compare.R's header:
#                             the port numbers must match the Rust engine to
#                             round-off), so they gate the BINDINGS, not the math --
#                             worth running when a port, the lowering, or the data
#                             changed, and largely redundant for a kernel-only change.
#   ./run.sh --oracles       ALSO refit R/lme4 and Julia/MixedModels. Needed when
#                             results/ is empty (it is gitignored, so a fresh clone
#                             has none), after --prep, or for a newly added rung.
#   ./run.sh --timings[=N]   ALSO time every fit that runs: N samples (default 4),
#                             warm-up discarded, median of the rest. N lives HERE,
#                             not in the engines -- they read VALIDATION_TIMINGS.
#                             Applies to whichever engines the flags above selected.
#                             `=N` must be attached; a bare `--timings N` would be
#                             ambiguous with the trailing dataset names. Meaningful
#                             ONLY on a locked machine (run `bench-l` first) -- treat
#                             it as a "did this get 10x slower" smoke check, not a
#                             measurement. Real speed work belongs in campaigns/.
#                             Records results/run_meta_<engine>.json (machine, git
#                             rev, no_turbo, pin) so summarize_timing.R can refuse to
#                             compare seconds fitted on two different boxes.
#   ./run.sh --prep          regenerate data_{empirical,simulated}/*.csv first (all five
#                             prep scripts: export_data.R for rungs 1-28,
#                             gen_weights_data.R for the prior-weights tier,
#                             gen_large_theta_data.R for the large-theta-hat rungs
#                             + the non-rung sim_binomial_zerosd fixture,
#                             gen_illcond_data.R for the two ill-conditioned LMM
#                             designs, which are goldens rather than rungs, and
#                             gen_scale_data.R for the scale-variation GLM goldens).
#                             IMPLIES --oracles: changed data invalidates the old
#                             R/Julia results.
#   ./run.sh --rust-tier2    ALSO run the crate's own cross-engine tier
#                             (`cargo test -p glmm --features oracle-tests`) first.
#   ./run.sh [flags] ds...   restrict every engine that runs to the named manifest
#                             datasets -- validated against manifest.json, unknown
#                             names fail loudly before anything is fit.
#
# A rung with no lme4 result JSON on disk is simply NOT LISTED by compare.R, which
# iterates the references it finds rather than the manifest. That is the intended
# degradation, not a skip to fix -- `--oracles` is how a rung joins the comparison.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREP=0
ORACLES=0
PORTS=0
TIMINGS=0
TIER2=0
# Sample count for --timings, warm-up included. 4 taken, first discarded, median of
# 3 -- enough to see a 10x regression, cheap enough that nobody is tempted to make
# the gate pay for it. This is THE definition: the engines carry no N_RUNS constant
# of their own any more, they read whatever lands in VALIDATION_TIMINGS.
TIMING_RUNS=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep) PREP=1; shift ;;
    --oracles) ORACLES=1; shift ;;
    --ports) PORTS=1; shift ;;
    --timings) TIMINGS=1; shift ;;
    # `=N` form only. `--timings N` would be ambiguous with the trailing dataset
    # names (./run.sh --timings cbpp), so the count must be attached.
    --timings=*) TIMINGS=1; TIMING_RUNS="${1#*=}"; shift ;;
    --rust-tier2) TIER2=1; shift ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
# --prep implies --oracles: regenerated data invalidates the frozen lme4/MixedModels
# results, so they must be refit alongside Rust (see header comment).
[[ "$PREP" == 1 ]] && ORACLES=1

# Positional args left after flag parsing are manifest dataset names. Validate against
# manifest.json's "name" fields with grep (jq is not assumed installed) so an unknown
# name fails loudly before any engine runs, rather than silently fitting nothing.
if [[ $# -gt 0 ]]; then
  for ds in "$@"; do
    grep -qF "\"name\": \"$ds\"" "$ROOT/manifest.json" \
      || { echo "unknown dataset: $ds (see validation/manifest.json)" >&2; exit 2; }
  done
  VALIDATION_ONLY="$(IFS=,; echo "$*")"
  export VALIDATION_ONLY
fi

# THE contract every engine implements, in five languages that cannot share code:
# VALIDATION_TIMINGS unset / "" / "0" means do not time, emit "timing": null.
# Otherwise it IS the sample count -- an integer >= 2, first sample discarded,
# median of the rest. Validated here so all five can trust the value; each engine
# still errors loudly on a malformed one, for the case where it is run by hand.
if [[ "$TIMINGS" == 1 ]]; then
  [[ "$TIMING_RUNS" =~ ^[0-9]+$ ]] && (( TIMING_RUNS >= 2 )) \
    || { echo "--timings=N needs an integer N >= 2 (got '$TIMING_RUNS'); N=2 keeps 1 sample after the warm-up discard" >&2; exit 2; }
  export VALIDATION_TIMINGS="$TIMING_RUNS"
fi

# Per-ENGINE run metadata for timed passes, mirroring campaigns/speed-grid/run.sh
# (which is where the no_turbo discipline was worked out) and memory/memory.sh's
# write_run_meta. Per engine, not per invocation: results/ legitimately holds legs
# fitted at different times on different boxes, so one run-level file would lie
# about the mix. Seconds do not transfer across machines -- `machine` is what lets
# summarize_timing.R refuse to put rows from two boxes in one comparison.
#
# Lives in results/ ITSELF, not results/<engine>_<suite>/: compare.R's read_engine()
# globs *.json in the per-suite dirs and immediately reads [["dataset"]] on each
# file, and unlike summarize_parallel.R it has no run_meta.json skip -- dropping one
# in there would break the gate.
write_run_meta() {
  local engine="$1" rev no_turbo
  rev="$(git -C "$ROOT/.." rev-parse HEAD 2>/dev/null || echo unknown)"
  # Recorded, never set -- clock locking is the user's `bench-l`/`bench-u`.
  no_turbo="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
  mkdir -p "$ROOT/results"
  printf '{"engine":"%s","machine":"%s","glmm_git_rev":"%s","n_runs":%s,"no_turbo":"%s","pin":"%s","started":"%s"}\n' \
    "$engine" "$(uname -n) $(uname -s)/$(uname -m)" "$rev" "$TIMING_RUNS" "$no_turbo" "${PIN:-none}" "$(date -Is)" \
    > "$ROOT/results/run_meta_${engine}.json"
  [[ "$no_turbo" == "1" ]] || echo \
    "   WARNING: clock NOT locked (no_turbo=$no_turbo) -- timings from this $engine pass are powersave noise; run bench-l first" >&2
}

# Oracles first: the Rust engine reads results/lme4_<suite>/<ds>.json for the
# reference grouping order, so lme4 must have run at least once in this tree.
ENGINES=(rust)
[[ "$PORTS" == 1 ]] && ENGINES=("${ENGINES[@]}" py glmm_r)
[[ "$ORACLES" == 1 ]] && ENGINES=(lme4 jl "${ENGINES[@]}")

# Pin TIMED fits to one P-core (cores 0-5 are the 5.3 GHz P-cores on this box; same
# core every run) so a locked-machine run isn't perturbed by the scheduler hopping
# onto a slower E-core (6-13) or LP-E core (14-15). No-op if taskset is absent, and
# NOT applied to an untimed run -- confining a pure correctness pass to one core buys
# nothing and only makes it slower. This does NOT lock the machine -- run `bench-l`
# yourself first, else the numbers are powersave noise regardless of pinning.
PIN=""
[[ "$TIMINGS" == 1 ]] && command -v taskset >/dev/null && PIN="taskset -c 1"

# Before the engines, not after: Tier 2 is the cheaper claim and needs no R or
# Julia, so a crate regression is reported without first paying for a full oracle
# refit. Unpinned -- it is a correctness gate, nothing here is timed.
if [[ "$TIER2" == 1 ]]; then
  echo ">> glmm Tier 2 (cross-engine tests)"
  cargo test --quiet --manifest-path "$ROOT/../Cargo.toml" -p glmm --features oracle-tests
fi

if [[ "$PREP" == 1 ]]; then
  echo ">> prep: regenerating committed data_{empirical,simulated}/*.csv"
  Rscript "$ROOT/prep/export_data.R"
  Rscript "$ROOT/prep/gen_weights_data.R"
  Rscript "$ROOT/prep/gen_large_theta_data.R"
  Rscript "$ROOT/prep/gen_illcond_data.R"
  Rscript "$ROOT/prep/gen_scale_data.R"
fi

for e in "${ENGINES[@]}"; do
  # Cleared by the skip branches below, so a missing julia/cargo/wheel/package does
  # not leave behind a run_meta claiming that engine was timed on this box.
  RAN=1
  case "$e" in
    lme4)
      echo ">> lme4 (R)"
      $PIN Rscript "$ROOT/engines/lme4.R" ;;
    jl)
      echo ">> MixedModels (Julia)"
      if ! command -v julia >/dev/null || [[ ! -f "$ROOT/Manifest.toml" ]]; then
        echo "   skipped: julia or the pinned env is missing (see README setup)" >&2; RAN=0
      else
        $PIN julia --project="$ROOT" "$ROOT/engines/mixedmodels.jl"
      fi ;;
    rust)
      echo ">> glmm (Rust) -> results/glmm_{empirical,simulated}/"
      if ! command -v cargo >/dev/null; then
        echo "   skipped: cargo not found" >&2; RAN=0
      else
        $PIN cargo run --quiet --release --manifest-path "$ROOT/../Cargo.toml" \
          -p validation --example validation_fit
      fi ;;
    py)
      echo ">> glmm python port -> results/glmm_python_*/"
      # The repo's own venv first (python/venv, where `maturin develop --release`
      # installs the wheel editable), else whatever python3 has glmm importable.
      # The wheel MUST be a --release build: a debug kernel would report the port's
      # "overhead" as a codegen artifact an order of magnitude too large.
      # dirname "$ROOT" (already absolute), not "$ROOT/..": a literal ".." in
      # sys.executable makes the venv's site.py warn about an unexpected sys.prefix.
      PY="$(dirname "$ROOT")/python/venv/bin/python"
      [[ -x "$PY" ]] || PY="$(command -v python3 || true)"
      if [[ -z "$PY" ]] || ! "$PY" -c 'import glmm' 2>/dev/null; then
        echo "   skipped: no python with the glmm wheel installed (see README setup)" >&2; RAN=0
      else
        $PIN "$PY" "$ROOT/engines/glmm_python.py"
      fi ;;
    glmm_r)
      echo ">> glmm R port -> results/glmm_r_*/"
      # Same kernel as the Rust/Python engines, reached through the fastglmm R
      # package (extendr wrapper). No venv step -- the package is installed in the
      # R library. Skip cleanly if it is not, so a machine without it still runs
      # the rest.
      if ! Rscript -e 'if (!requireNamespace("fastglmm", quietly=TRUE)) quit(status=1)' 2>/dev/null; then
        echo "   skipped: fastglmm R package not installed (see README setup)" >&2; RAN=0
      else
        $PIN Rscript "$ROOT/engines/glmm_r.R"
      fi ;;
    *)
      echo "unknown engine: $e" >&2; exit 2 ;;
  esac
  # run_meta and the timings it describes move together. An untimed pass has just
  # rewritten this engine's results with "timing": null, so a meta left behind from
  # an earlier --timings run would advertise provenance for numbers that no longer
  # exist. Plain rm, not trash-put: this is a regenerable file in a gitignored dir,
  # and the harness must not require trash-cli to run.
  if [[ "$RAN" == 1 ]]; then
    if [[ "$TIMINGS" == 1 ]]; then
      write_run_meta "$e"
    else
      rm -f "$ROOT/results/run_meta_${e}.json"
    fi
  fi
done

echo ">> compare"
Rscript "$ROOT/compare.R"
