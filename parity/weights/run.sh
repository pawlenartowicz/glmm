#!/usr/bin/env bash
# WEIGHTS parity suite runner -- FitOptions.weights against the live oracles
# (docs/GLMM/2026-07-10-weights-parity-harness-design.md). Standalone: NOT wired
# into the main parity/run.sh gate; for 0.1.0 both suites must be green.
#
# Thin wrapper: sets the 5 suite-specific overrides (suite dir, prep script,
# results/README/manifest display text -- everything that differs from the main
# suite) and execs the shared ../run.sh, which does the actual work. Flags and
# dataset-name validation are handled there; see its header for usage.
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARITY="$(dirname "$SUITE")"   # main parity/ -- shared run.sh + engine scripts

export PARITY_SUITE_DIR="$SUITE"
export PARITY_PREP_SCRIPT="gen_weights_data.R"
export PARITY_PREP_DESC="data_simulated/*.csv"
export PARITY_RESULTS_DESC="glmm_simulated/"
export PARITY_README_HINT="../README.md setup"
export PARITY_MANIFEST_HINT="parity/weights/manifest.json"

exec "$PARITY/run.sh" "$@"
