#!/usr/bin/env bash
# Run the fit engines over manifest datasets, then the agreement check.
# Adding Rust later is ONE list entry (`rust`) plus oracle/fit.rs -- no restructuring.
#
#   ./run.sh                 fit glmm (Rust) only + compare against the EXISTING
#                             results/lme4_*/mixedmodels_* JSONs on disk (fast default;
#                             R/lme4 and Julia/MixedModels are NOT refit -- those results
#                             don't change run to run, so paying their cost every time is
#                             wasted). Never touches data_{empirical,simulated}/.
#   ./run.sh --oracles       refit ALL THREE engines (R, Julia, Rust) + compare -- use when
#                             regenerating the oracle itself (new dataset, new machine,
#                             tolerance work).
#   ./run.sh --prep          regenerate the committed data_{empirical,simulated}/*.csv
#                             first. IMPLIES --oracles: changed data invalidates the old
#                             R/Julia results, so a --prep run always refits all three
#                             (silently upgrading --prep alone to also refit lme4/MM would
#                             be surprising; --prep --oracles is accepted as the same thing).
#   ./run.sh [flags] ds...   restrict every engine that runs to the named manifest
#                             datasets (e.g. `./run.sh cbpp`, `./run.sh --oracles cbpp
#                             sleepstudy`) -- validated against manifest.json, unknown
#                             names fail loudly before anything is fit.
set -euo pipefail
PARITY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREP=0
ORACLES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep) PREP=1; shift ;;
    --oracles) ORACLES=1; shift ;;
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
    grep -qF "\"name\": \"$ds\"" "$PARITY/manifest.json" \
      || { echo "unknown dataset: $ds (see parity/manifest.json)" >&2; exit 2; }
  done
  PARITY_ONLY="$(IFS=,; echo "$*")"
  export PARITY_ONLY
fi

ENGINES=(rust)
[[ "$ORACLES" == 1 ]] && ENGINES=(R jl rust)

# Pin the timed fits to one P-core (cores 0-5 are the 5.3 GHz P-cores on this box;
# same core every run) so a locked-machine run isn't perturbed by the scheduler
# hopping onto a slower E-core (6-13) or LP-E core (14-15). No-op if taskset is
# absent. This does NOT lock the machine -- run `bench-l` yourself first, else the
# numbers are powersave noise regardless of pinning. compare.R is untimed, unpinned.
PIN=""
command -v taskset >/dev/null && PIN="taskset -c 1"

if [[ "$PREP" == 1 ]]; then
  echo ">> prep: regenerating committed data_{empirical,simulated}/*.csv from lme4"
  Rscript "$PARITY/prep/export_data.R"
fi

for e in "${ENGINES[@]}"; do
  case "$e" in
    R)
      echo ">> lme4 (R)"
      $PIN Rscript "$PARITY/oracle/fit.R" ;;
    jl)
      echo ">> MixedModels (Julia)"
      if ! command -v julia >/dev/null || [[ ! -f "$PARITY/Manifest.toml" ]]; then
        echo "   skipped: julia or the pinned env is missing (see README setup)" >&2
      else
        $PIN julia --project="$PARITY" "$PARITY/oracle/fit.jl"
      fi ;;
    rust)
      echo ">> glmm (Rust) -> results/glmm_{empirical,simulated}/"
      if ! command -v cargo >/dev/null; then
        echo "   skipped: cargo not found" >&2
      else
        $PIN cargo run --quiet --release --manifest-path "$PARITY/../Cargo.toml" \
          --example parity_fit
      fi ;;
    *)
      echo "unknown engine: $e" >&2; exit 2 ;;
  esac
done

echo ">> compare"
Rscript "$PARITY/compare.R"
