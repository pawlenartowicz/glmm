#!/usr/bin/env bash
# WEIGHTS parity suite runner -- FitOptions.weights against the live oracles
# (docs/GLMM/2026-07-10-weights-parity-harness-design.md). Standalone: NOT wired
# into the main parity/run.sh gate; for 0.1.0 both suites must be green.
#
# Mirrors the main runner's interface exactly. It reuses the SHARED engine
# scripts (../oracle/fit.{R,jl,rs}, ../compare.R) by exporting PARITY_SUITE_DIR
# to this directory, under which those scripts resolve manifest.json,
# data_simulated/ and results/.
#
#   ./run.sh                 fit glmm (Rust) only + compare against the EXISTING
#                             results/lme4_*/mixedmodels_* JSONs on disk. Requires the
#                             committed R results (fit.rs reads each rung's lme4 JSON
#                             for varcomp grouping order, fixed-only rungs included).
#   ./run.sh --oracles       refit ALL THREE engines (R, Julia, Rust) + compare.
#   ./run.sh --prep          regenerate the committed data_simulated/*.csv first.
#                             IMPLIES --oracles: changed data invalidates the old
#                             R/Julia results.
#   ./run.sh [flags] ds...   restrict every engine that runs to the named manifest
#                             datasets -- validated against weights/manifest.json,
#                             unknown names fail loudly (exit 2) before anything is fit.
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARITY="$(dirname "$SUITE")"   # main parity/ -- shared scripts + pinned Julia env
export PARITY_SUITE_DIR="$SUITE"

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
# --prep implies --oracles: regenerated data invalidates the frozen references.
[[ "$PREP" == 1 ]] && ORACLES=1

# Positional args are manifest dataset names (grep, not jq -- mirrors the main
# runner): unknown names fail loudly before any engine runs.
if [[ $# -gt 0 ]]; then
  for ds in "$@"; do
    grep -qF "\"name\": \"$ds\"" "$SUITE/manifest.json" \
      || { echo "unknown dataset: $ds (see parity/weights/manifest.json)" >&2; exit 2; }
  done
  PARITY_ONLY="$(IFS=,; echo "$*")"
  export PARITY_ONLY
fi

ENGINES=(rust)
[[ "$ORACLES" == 1 ]] && ENGINES=(R jl rust)

# Same pinning rationale as the main runner: one P-core, same core every run;
# numbers are meaningful only on a clock-locked machine (user runs bench-l).
PIN=""
command -v taskset >/dev/null && PIN="taskset -c 1"

if [[ "$PREP" == 1 ]]; then
  echo ">> prep: regenerating committed data_simulated/*.csv"
  Rscript "$SUITE/prep/gen_weights_data.R"
fi

for e in "${ENGINES[@]}"; do
  case "$e" in
    R)
      echo ">> lm/glm/lme4 (R)"
      $PIN Rscript "$PARITY/oracle/fit.R" ;;
    jl)
      echo ">> MixedModels (Julia)"
      if ! command -v julia >/dev/null || [[ ! -f "$PARITY/Manifest.toml" ]]; then
        echo "   skipped: julia or the pinned env is missing (see ../README.md setup)" >&2
      else
        $PIN julia --project="$PARITY" "$PARITY/oracle/fit.jl"
      fi ;;
    rust)
      echo ">> glmm (Rust) -> results/glmm_simulated/"
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
