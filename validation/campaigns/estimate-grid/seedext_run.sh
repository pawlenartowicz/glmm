#!/usr/bin/env bash
# Seed-extension study driver (lme4 issue #998 / tolPwrss question): three
# passes over manifest_seedext.json (945 cells from seedext_gen.R) in two
# parallel pinned streams:
#   stream A (core 1): lme4 tolPwrss=1e-7 (glmer's default), then glmm
#   stream B (core 2): lme4 tolPwrss=1e-11
# Untimed correctness run -- no clock lock needed. Existing campaign results
# are untouched (new GRID_OUT files, new tags).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GRID_MANIFEST="$HERE/manifest_seedext.json"
export GRID_TODO="$HERE/seedext_todo.txt"

(
  GRID_PIN_CORE=1 GRID_TOLPWRSS=1e-7 \
    GRID_OUT="$HERE/results/lme4_seedext_default.jsonl" \
    "$HERE/../speed-grid/run.sh" lme4 seedext-default
  GRID_PIN_CORE=1 GRID_OUT="$HERE/results/glmm_seedext.jsonl" \
    "$HERE/../speed-grid/run.sh" glmm seedext
) > "$HERE/results/seedext_streamA.log" 2>&1 &
A=$!
(
  GRID_PIN_CORE=2 GRID_TOLPWRSS=1e-11 \
    GRID_OUT="$HERE/results/lme4_seedext_t11.jsonl" \
    "$HERE/../speed-grid/run.sh" lme4 seedext-t11
) > "$HERE/results/seedext_streamB.log" 2>&1 &
B=$!
RC=0
wait "$A" || RC=$?
wait "$B" || RC=$?
echo "seedext passes done (rc=$RC)"
exit "$RC"
