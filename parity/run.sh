#!/usr/bin/env bash
# Run every present engine's fit over all manifest datasets, then the agreement check.
# Adding Rust later is ONE list entry (`rust`) plus oracle/fit.rs -- no restructuring.
#
#   ./run.sh           fit all engines + compare (ordinary run; never touches data/)
#   ./run.sh --prep    regenerate the committed data/*.csv first, then fit + compare
set -euo pipefail
PARITY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENGINES=(R jl rust)

# Pin the timed fits to one P-core (cores 0-5 are the 5.3 GHz P-cores on this box;
# same core every run) so a locked-machine run isn't perturbed by the scheduler
# hopping onto a slower E-core (6-13) or LP-E core (14-15). No-op if taskset is
# absent. This does NOT lock the machine -- run `bench-l` yourself first, else the
# numbers are powersave noise regardless of pinning. compare.R is untimed, unpinned.
PIN=""
command -v taskset >/dev/null && PIN="taskset -c 1"

if [[ "${1:-}" == "--prep" ]]; then
  echo ">> prep: regenerating committed data/*.csv from lme4"
  Rscript "$PARITY/prep/export_data.R"
  shift || true
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
      echo ">> glmm (Rust) -> results/glmm/"
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
