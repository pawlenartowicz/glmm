#!/usr/bin/env bash
# Optimizer-grid campaign driver: runs one engine pass under a per-fit wall
# watchdog with kill-and-resume (native optimizer calls can't be interrupted
# in-process — design doc "Pathology guards"). The engine appends+flushes one
# JSONL line per fit, so "output mtime is stale" == "the current fit exceeded
# its wall budget": kill the engine, write a status=timeout record for the
# first not-yet-done cell, relaunch (the runner skips finished cells).
#
# CLOCK: the user locks/unlocks (bench-l / bench-u) — this script only RECORDS
# the state (no_turbo) into run_meta so unlocked timings can be excluded later.
#
#   ./run_grid.sh glmm|mixedmodels|lme4 <pass-tag> [timeout-seconds]
# Timeouts default per design: glmm/mixedmodels 10 s, lme4 120 s; B passes
# override with the third arg (60).
set -euo pipefail
PARITY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${1:?engine}"; TAG="${2:?pass tag}"; TIMEOUT="${3:-}"
MANIFEST="${GRID_MANIFEST:-$PARITY/manifest_grid.json}"
OUT="${GRID_OUT:-$PARITY/results/grid/${ENGINE}_${TAG}.jsonl}"
mkdir -p "$(dirname "$OUT")"
export GRID_OUT="$OUT" GRID_MANIFEST="$MANIFEST" GRID_CONFIG_TAG="${GRID_CONFIG_TAG:-$TAG}"

case "$ENGINE" in
  glmm)        CMD=(cargo run --quiet --release --manifest-path "$PARITY/../Cargo.toml" --example grid_fit)
               TIMEOUT="${TIMEOUT:-10}" ;;
  mixedmodels) CMD=(julia --project="$PARITY" "$PARITY/oracle/grid_fit.jl")
               # 2x glmm's default: grid_fit.jl double-fits each cell (JIT
               # warm-up + timed fit) — callers passing an explicit budget
               # must double it the same way (run_study_a.sh does)
               TIMEOUT="${TIMEOUT:-20}" ;;
  lme4)        CMD=(Rscript "$PARITY/oracle/grid_fit.R")
               TIMEOUT="${TIMEOUT:-120}" ;;
  *) echo "unknown engine: $ENGINE" >&2; exit 2 ;;
esac
PIN=""; command -v taskset >/dev/null && PIN="taskset -c 1"

# per-launch startup grace: engine load writes nothing (Julia pkg load +
# first-fit JIT far exceeds the 10 s per-fit budget) — until this launch
# appends its first line, judge staleness against GRACE instead of TIMEOUT.
# Override with GRID_STARTUP_GRACE (the forced-timeout smoke test sets 0).
GRACE="${GRID_STARTUP_GRACE:-$((TIMEOUT + 180))}"

# glmm: compile OUTSIDE the watchdog — a first build takes minutes with no
# output writes, which would read as a per-fit timeout and kill the compiler.
[ "$ENGINE" = "glmm" ] && cargo build --quiet --release \
  --manifest-path "$PARITY/../Cargo.toml" --example grid_fit

# clock state into run meta (recorded, never set — user's bench-l/bench-u)
NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "?")
printf '{"engine":"%s","tag":"%s","timeout_s":%s,"no_turbo":"%s","started":"%s"}\n' \
  "$ENGINE" "$TAG" "$TIMEOUT" "$NO_TURBO" "$(date -Is)" \
  > "$PARITY/results/grid/run_meta_${ENGINE}_${TAG}.json"
[ "$NO_TURBO" = "1" ] || echo "WARNING: clock NOT locked (no_turbo=$NO_TURBO) — timings from this pass must be excluded from timing aggregates" >&2

# cell universe (lme4 runs only its GRID_TODO subset)
if [ "$ENGINE" = "lme4" ]; then
  : "${GRID_TODO:?lme4 pass needs GRID_TODO (from analyze_grid.R)}"
  export GRID_TODO
  mapfile -t ALL < "$GRID_TODO"
else
  mapfile -t ALL < <(jq -r '.cells[].case_id' "$MANIFEST")
fi

next_missing() {
  local done
  # -R + fromjson?: kill -9 can truncate the final line — skip it rather than
  # fail the whole scan (which would re-flag every finished cell as missing)
  done=$(jq -rR 'fromjson? | .case_id' "$OUT" 2>/dev/null | sort -u) || done=""
  for c in "${ALL[@]}"; do
    grep -qxF "$c" <<< "$done" || { echo "$c"; return 0; }
  done
  return 1
}

while MISSING=$(next_missing); do
  echo ">> $ENGINE/$TAG: next cell $MISSING ($(jq -rR 'fromjson? | .case_id' "$OUT" 2>/dev/null | sort -u | wc -l)/${#ALL[@]} done)"
  $PIN "${CMD[@]}" &
  ENGPID=$!
  touch "$OUT"
  LINES0=$(wc -l < "$OUT")
  while kill -0 "$ENGPID" 2>/dev/null; do
    sleep 1
    if (( $(wc -l < "$OUT") == LINES0 )); then EFF=$GRACE; else EFF=$TIMEOUT; fi
    NOW=$(date +%s); MT=$(stat -c %Y "$OUT")
    if (( NOW - MT > EFF )); then
      echo ">> timeout on $(next_missing || echo '?') — killing $ENGINE" >&2
      kill -9 "$ENGPID" 2>/dev/null || true
      wait "$ENGPID" 2>/dev/null || true
      CELL=$(next_missing || true)
      if [ -n "${CELL:-}" ]; then
        SEED=$(jq -r --arg c "$CELL" '.cells[] | select(.case_id==$c).seed' "$MANIFEST")
        printf '{"case_id":"%s","seed":%s,"engine":"%s","optimizer":"","config_tag":"%s","n_eval":0,"converged":false,"singular":false,"deviance":null,"beta":[],"se":[],"status":"timeout","wall_seconds":%s}\n' \
          "$CELL" "${SEED:-0}" "$ENGINE" "$GRID_CONFIG_TAG" "$TIMEOUT" >> "$OUT"
      fi
      break
    fi
  done
  wait "$ENGPID" 2>/dev/null || true
  # engine exited on its own with cells remaining and no progress ⇒ record an
  # engine-fail for the blocking cell so the loop can't spin forever
  if CELL=$(next_missing); then
    if [ "$CELL" = "${LAST_STUCK:-}" ]; then
      SEED=$(jq -r --arg c "$CELL" '.cells[] | select(.case_id==$c).seed' "$MANIFEST")
      printf '{"case_id":"%s","seed":%s,"engine":"%s","optimizer":"","config_tag":"%s","n_eval":0,"converged":false,"singular":false,"deviance":null,"beta":[],"se":[],"status":"engine-fail","wall_seconds":0}\n' \
        "$CELL" "${SEED:-0}" "$ENGINE" "$GRID_CONFIG_TAG" >> "$OUT"
      LAST_STUCK=""
    else
      LAST_STUCK="$CELL"
    fi
  fi
done
echo ">> $ENGINE/$TAG complete: ${#ALL[@]} cells in $OUT"
