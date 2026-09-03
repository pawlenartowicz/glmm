#!/usr/bin/env bash
# Optimizer-grid campaign driver: runs one engine pass under a per-cell wall
# watchdog with kill-and-resume (native optimizer calls can't be interrupted
# in-process — design doc "Pathology guards"). The engine appends+flushes one
# JSONL line per fit, so "output mtime is stale" == "the current cell exceeded
# its wall budget": kill the engine, write a status=timeout record for the
# first not-yet-done cell, relaunch (the runner skips finished cells).
#
# CLOCK: the user locks/unlocks (bench-l / bench-u) — this script only RECORDS
# the state (no_turbo) into run_meta so unlocked timings can be excluded later.
#
#   ./run.sh glmm|mixedmodels|lme4 <pass-tag> [budget-seconds]
# Budget is per-cell (not per-fit), 240 s by default for all three engines,
# and is not scaled by fits-per-cell. glmm treats it as a soft budget:
# fit_cell (fit.rs) decides predictively whether to start another fit and
# reports whatever finished inside it, so glmm's watchdog TIMEOUT is
# budget+60s of hang backstop, not the enforcement. MixedModels.jl and lme4
# have no usable partial result (JIT warm-up wall means nothing; lme4 does a
# single fit with no warm-up at all), so for them TIMEOUT equals the budget
# exactly and the watchdog IS the enforcement — an overrunning cell is killed
# and recorded as timeout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ENGINE="${1:?engine}"; TAG="${2:?pass tag}"; BUDGET="${3:-240}"
MANIFEST="${GRID_MANIFEST:-$HERE/manifest.json}"
OUT="${GRID_OUT:-$HERE/results/${ENGINE}_${TAG}.jsonl}"
mkdir -p "$(dirname "$OUT")"
export GRID_OUT="$OUT" GRID_MANIFEST="$MANIFEST" GRID_CONFIG_TAG="${GRID_CONFIG_TAG:-$TAG}"

case "$ENGINE" in
  glmm)        FEATFLAGS=()
               # Counter pass: build the driver against glmm's `counters`
               # feature so the per-cell record carries the stage split, the
               # shrink counts, the PIRLS histogram and the AGQ node cost.
               # Off by default — rationale on the `counters` feature in
               # validation/Cargo.toml.
               [ -n "${GRID_COUNTERS:-}" ] && FEATFLAGS=(--features validation/counters)
               CMD=(cargo run --quiet --release --manifest-path "$ROOT/../Cargo.toml" -p validation --example grid_fit "${FEATFLAGS[@]}")
               # Soft budget, enforced inside fit.rs (fit_cell reads
               # GRID_CELL_BUDGET). TIMEOUT here is a hang backstop only —
               # budget + 60s of slack for marshalling/process start
               # (measured ~0.1s, so the margin is generous on purpose).
               TIMEOUT=$((BUDGET + 60))
               export GRID_CELL_BUDGET="$BUDGET"
               # One cell per launch (fit.rs honors this — the two sites
               # change together): per-cell walls were process-context
               # dependent (2-10x swings on sub-100ms cells) when hundreds of
               # cells ran back to back in one process. A fresh process per
               # cell makes the timing context uniform. Julia/R keep the
               # in-process protocol — their per-launch startup/JIT cost
               # dwarfs a per-fit budget.
               export GRID_ONE_CELL=1 ;;
  mixedmodels) CMD=(julia --project="$ROOT" "$HERE/fit.jl")
               # Hard kill: fit.jl's JIT warm-up wall means nothing as a
               # partial result, so the watchdog IS the enforcement.
               TIMEOUT="$BUDGET" ;;
  lme4)        CMD=(Rscript "$HERE/fit.R")
               # Hard kill: a single fit with no warm-up has no partial
               # result to fall back on either.
               TIMEOUT="$BUDGET" ;;
  *) echo "unknown engine: $ENGINE" >&2; exit 2 ;;
esac
PIN=""; command -v taskset >/dev/null && PIN="taskset -c ${GRID_PIN_CORE:-1}"

# per-launch startup grace: engine load writes nothing (Julia pkg load +
# first-fit JIT can exceed the per-cell budget) — until this launch appends
# its first line, judge staleness against GRACE instead of TIMEOUT.
# Override with GRID_STARTUP_GRACE (the forced-timeout smoke test sets 0).
GRACE="${GRID_STARTUP_GRACE:-$((TIMEOUT + 180))}"

# glmm: compile OUTSIDE the watchdog — a first build takes minutes with no
# output writes, which would read as a per-fit timeout and kill the compiler.
[ "$ENGINE" = "glmm" ] && cargo build --quiet --release \
  --manifest-path "$ROOT/../Cargo.toml" -p validation --example grid_fit "${FEATFLAGS[@]}"

# clock state into run meta (recorded, never set — user's bench-l/bench-u)
NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "?")
printf '{"engine":"%s","tag":"%s","timeout_s":%s,"no_turbo":"%s","started":"%s"}\n' \
  "$ENGINE" "$TAG" "$TIMEOUT" "$NO_TURBO" "$(date -Is)" \
  > "$(dirname "$OUT")/run_meta_${ENGINE}_${TAG}.json"
[ "$NO_TURBO" = "1" ] || echo "WARNING: clock NOT locked (no_turbo=$NO_TURBO) — timings from this pass must be excluded from timing aggregates" >&2

# cell universe (lme4 runs only its GRID_TODO subset; other engines respect
# GRID_ONLY the same way fit.jl/.rs do — otherwise the watchdog treats
# every cell GRID_ONLY excluded as still "missing" and relaunches forever)
if [ "$ENGINE" = "lme4" ]; then
  : "${GRID_TODO:?lme4 pass needs GRID_TODO (from analyze.R)}"
  export GRID_TODO
  mapfile -t ALL < "$GRID_TODO"
elif [ -n "${GRID_ONLY:-}" ]; then
  mapfile -t ALL < <(tr ',' '\n' <<< "$GRID_ONLY")
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
  KILLED=0
  while kill -0 "$ENGPID" 2>/dev/null; do
    sleep 1
    # One-cell-per-launch engines write nothing until the cell is done, so the
    # first-line test can never fire for them and the startup grace would apply
    # for the whole run (bugtracker 30). Their launch-to-write time IS the cell
    # time, so hold them to TIMEOUT from the start. The grace is for engines that
    # amortise a slow startup across many cells in one process (Julia).
    if [ -n "${GRID_ONE_CELL:-}" ]; then EFF=$TIMEOUT
    elif (( $(wc -l < "$OUT") == LINES0 )); then EFF=$GRACE
    else EFF=$TIMEOUT; fi
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
      KILLED=1
      break
    fi
  done
  RC=0; wait "$ENGPID" 2>/dev/null || RC=$?
  # The engine exited on its own with cells remaining. Its exit code decides
  # what that means, and conflating the two poisoned a whole pass once
  # (2026-08-06 npt-mid: 24 consecutive cells, one of them a 4 ms 300-row fit,
  # recorded engine-fail with n_eval=0 — the launches never ran).
  #   RC != 0 — the LAUNCH failed before reaching the cell: build error, OOM
  #     kill, panic at startup. Nothing whatever is known about the cell, so
  #     writing a result for it fabricates data. Two strikes, then stop.
  #   RC == 0 — the engine ran and declined the cell without writing it. That
  #     IS a cell-level failure and is recorded as one, so the loop can't spin.
  # Skipped after a watchdog kill: the timeout record is already written, and
  # next_missing now names the FOLLOWING cell, which nothing has tried yet.
  if [ "$KILLED" = 0 ] && CELL=$(next_missing); then
    if [ "$CELL" != "${LAST_STUCK:-}" ]; then
      LAST_STUCK="$CELL"
    elif [ "$RC" != 0 ]; then
      echo ">> $ENGINE launch exited $RC twice with no output on $CELL — aborting" >&2
      echo ">> (build failure or OOM; nothing is recorded for this cell)" >&2
      exit 1
    else
      SEED=$(jq -r --arg c "$CELL" '.cells[] | select(.case_id==$c).seed' "$MANIFEST")
      printf '{"case_id":"%s","seed":%s,"engine":"%s","optimizer":"","config_tag":"%s","n_eval":0,"converged":false,"singular":false,"deviance":null,"beta":[],"se":[],"status":"engine-fail","wall_seconds":0}\n' \
        "$CELL" "${SEED:-0}" "$ENGINE" "$GRID_CONFIG_TAG" >> "$OUT"
      LAST_STUCK=""
    fi
  fi
done
echo ">> $ENGINE/$TAG complete: ${#ALL[@]} cells in $OUT"
