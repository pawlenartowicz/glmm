#!/usr/bin/env bash
# Run every present engine's fit over all manifest datasets, then the agreement check.
# Adding Rust later is ONE list entry (`rust`) plus oracle/fit.rs -- no restructuring.
#
#   ./run.sh           fit all engines + compare (ordinary run; never touches data/)
#   ./run.sh --prep    regenerate the committed data/*.csv first, then fit + compare
set -euo pipefail
PARITY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENGINES=(R jl)   # <- add "rust" here when oracle/fit.rs lands

if [[ "${1:-}" == "--prep" ]]; then
  echo ">> prep: regenerating committed data/*.csv from lme4"
  Rscript "$PARITY/prep/export_data.R"
  shift || true
fi

for e in "${ENGINES[@]}"; do
  case "$e" in
    R)
      echo ">> lme4 (R)"
      Rscript "$PARITY/oracle/fit.R" ;;
    jl)
      echo ">> MixedModels (Julia)"
      if ! command -v julia >/dev/null || [[ ! -f "$PARITY/Manifest.toml" ]]; then
        echo "   skipped: julia or the pinned env is missing (see README setup)" >&2
      else
        julia --project="$PARITY" "$PARITY/oracle/fit.jl"
      fi ;;
    rust)
      echo ">> glmm (Rust): oracle/fit.rs not present yet -- deferred (design 7)" ;;
    *)
      echo "unknown engine: $e" >&2; exit 2 ;;
  esac
done

echo ">> compare"
Rscript "$PARITY/compare.R"
