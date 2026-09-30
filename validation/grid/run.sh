#!/usr/bin/env bash
# The one runner for every accuracy-grid engine.
#
#   ./run.sh <engine> [--fast] [--timed[=N]] [--jobs N] [--label TEXT] [--keep] [cell ...]
#
#   engine   glmm | glmm_wasm | glmm_python | glmm_r | lme4 | glmmtmb | glmmadaptive | mixedmodels
#            (glmm_wasm: the glmm engine built for wasm32-wasip1 with simd128, run
#            under wasmtime)
#   --fast   restrict to the cells tagged `fast` in manifest.json
#   --timed[=N]  time every fit: N samples (default 4), first discarded, median of
#            the rest. IMPLIES --fast and --jobs=1. Records no_turbo and REFUSES to
#            start when it reads 0 unless --timed-unlocked is given, in which case
#            the run is marked unlocked in run_meta and summarize_timing.R warns.
#            The user locks the machine (bench-l); this script only records the state.
#   --jobs N split the cell list into N chunks and run N workers, worker i pinned
#            with `taskset -c i`. P-cores on this box are 0-5, so N > 6 is refused.
#   --label  free text purpose of the run ("release 0.4.0 baseline"), stored in
#            run_meta and slugged into the directory name.
#   --keep   a glmm / glmm_wasm / glmm_python / glmm_r run lands in runs/<engine>/ instead of
#            runs/<engine>/scratch/. Oracle runs are always kept.
#   cell ... restrict to the named cells; validated against manifest.json, so an
#            unknown name fails loudly before anything is fit.
#
# FLAG PARSING. Flags must come BEFORE the trailing cell names; the parser stops
# at the first non-flag argument. Given that,
# `--jobs N` and `--jobs=N` are both unambiguous and both accepted. `--timed` is
# the exception: it is optional-valued, so a bare `--timed N` WOULD be ambiguous
# with a cell name and the count must be attached -- `--timed` alone means the
# default 4, `--timed=6` means 6.
#
# RESUME. Each worker runs its cells under a per-cell watchdog: the engine
# appends and flushes one JSONL line per fit, so "the output's mtime is stale"
# means "the current cell blew its budget" -- kill the engine, record a timeout
# for the first cell still missing, relaunch (every engine skips cells already in
# its output). That is resume WITHIN one invocation and it is automatic.
# A whole invocation that dies (Ctrl-C, reboot) is NOT resumed: its part files
# stay where they are and you simply launch the command again, which fits a fresh
# run directory from scratch. Nothing is overwritten either way.
set -euo pipefail
GRID="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The crate root: grid/ sits in validation/, which sits in the crate.
ROOT="$(cd "$GRID/../.." && pwd)"
command -v jq >/dev/null || { echo "run.sh needs jq" >&2; exit 2; }

ENGINE="${1:-}"
[[ -n "$ENGINE" ]] || { echo "usage: ./run.sh <engine> [flags] [cell ...]" >&2; exit 2; }
shift

FAST=0
TIMED_N=""
TIMED_UNLOCKED=0
JOBS=1
LABEL=""
KEEP=0
# Per-cell watchdog budget in seconds. Used ONLY by the watchdog below and never
# exported: no engine caps its own fits. 600 rather than the speed
# campaign's 240 because the grid's 30000-row wide cells are the slowest fits in
# the corpus and a 240 s kill would record every one of them as a timeout.
BUDGET="${GRID_WATCHDOG:-600}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) FAST=1; shift ;;
    --timed) TIMED_N=4; shift ;;
    --timed=*) TIMED_N="${1#*=}"; shift ;;
    --timed-unlocked) TIMED_UNLOCKED=1; shift ;;
    --jobs) JOBS="${2:?--jobs needs a count}"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --label) LABEL="${2:?--label needs text}"; shift 2 ;;
    --label=*) LABEL="${1#*=}"; shift ;;
    --keep) KEEP=1; shift ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
CELL_ARGS=("$@")

# --timed implies --fast and --jobs 1: it raises the SAME flag --fast raises, so
# there is one path to SUBSET=fast rather than two.
[[ -n "$TIMED_N" ]] && { FAST=1; JOBS=1; }

# SUBSET is THREE-VALUED and derived HERE, after parsing, never assigned by a
# flag handler. The narrowest claim wins:
#   positional cell names -> "cells"
#   --fast / --timed      -> "fast"
#   neither               -> "full"
# compare.R accepts only "full" as an oracle reference, so a smoke, probe or
# pilot run is structurally unusable as one and needs no cleanup afterwards.
# THIS IS THE ONLY PLACE SUBSET IS WRITTEN.
if   (( ${#CELL_ARGS[@]} )); then SUBSET=cells
elif [[ "$FAST" == 1 ]];     then SUBSET=fast
else                              SUBSET=full
fi
# The array run_meta records as `cell_list`: the positional names, or [].
if (( ${#CELL_ARGS[@]} )); then
  CELL_LIST_JSON="$(printf '%s\n' "${CELL_ARGS[@]}" | jq -R . | jq -s .)"
else
  CELL_LIST_JSON='[]'
fi

[[ "$JOBS" =~ ^[0-9]+$ ]] && (( JOBS >= 1 && JOBS <= 6 )) \
  || { echo "--jobs must be 1..6 (P-cores are 0-5 on this box)" >&2; exit 2; }
if [[ -n "$TIMED_N" ]]; then
  [[ "$TIMED_N" =~ ^[0-9]+$ ]] && (( TIMED_N >= 2 )) \
    || { echo "--timed=N needs an integer N >= 2; N=2 keeps 1 sample after the warm-up discard" >&2; exit 2; }
  NO_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
  if [[ "$NO_TURBO" != "1" && "$TIMED_UNLOCKED" != 1 ]]; then
    echo "REFUSING a timed run: no_turbo=$NO_TURBO (clock not locked). Run bench-l first," >&2
    echo "or pass --timed-unlocked to record the run as unlocked." >&2
    exit 2
  fi
fi

# The spelling the manifest's `oracles` array uses. The empty string means "no
# filter": glmm and its two ports fit every cell, and `oracles` lists only the
# external oracles. This case is the ONLY engine-name check -- the two cases
# further down (the version probe, the command) mirror its engine list and must
# change with it.
case "$ENGINE" in
  lme4)         ENGINE_MANIFEST_NAME=lme4 ;;
  glmmtmb)      ENGINE_MANIFEST_NAME=glmmTMB ;;
  glmmadaptive) ENGINE_MANIFEST_NAME=GLMMadaptive ;;
  mixedmodels)  ENGINE_MANIFEST_NAME=MixedModels ;;
  glmm|glmm_wasm|glmm_python|glmm_r) ENGINE_MANIFEST_NAME="" ;;
  *) echo "unknown engine: $ENGINE" >&2; exit 2 ;;
esac

PIN_CORES=none
command -v taskset >/dev/null && PIN_CORES="0-$((JOBS - 1))"
[[ "$PIN_CORES" == none ]] && echo ">> taskset not found -- workers run unpinned" >&2

# ---- cell selection --------------------------------------------------------
# SELECTED is built in one deterministic order and is the ONLY input to the
# chunker besides JOBS. Order: n_obs DESCENDING, then n_theta DESCENDING, then
# cell id ASCENDING -- a total order, so a cell's rank never depends on which
# cells ran before it.
#
# `sort_by` on an array compares element-wise, and the negated numeric keys turn
# "descending" into jq's ascending sort without a second pass.
# THE `fast` TAG FILTER APPLIES ONLY WHEN SUBSET == "fast". Both "full" and
# "cells" start from the whole manifest; a cell-restricted run narrows the list
# AFTERWARDS by intersecting with CELL_ARGS (below), so naming a cell that
# carries no `fast` tag works. Without the third arm here, a run restricted to
# one non-fast cell would select nothing -- SUBSET is "cells", and a two-arm
# filter would demand the `fast` tag of anything that is not "full".
select_cells() {
  jq -r --arg subset "$SUBSET" --arg eng "$ENGINE_MANIFEST_NAME" '
    [ .cells[]
      | select(($subset == "full") or ($subset == "cells") or (.tags | index("fast")))
      | select(.oracles | index($eng) or ($eng == ""))
      | {c: .cell, a: (-(.n_obs)), b: (-(.n_theta // 0))} ]
    | sort_by([.a, .b, .c]) | .[].c' "$GRID/manifest.json"
}

if (( ${#CELL_ARGS[@]} )); then
  ALL_IDS="$(jq -r '.cells[].cell' "$GRID/manifest.json")"
  for c in "${CELL_ARGS[@]}"; do
    grep -qxF "$c" <<< "$ALL_IDS" || { echo "unknown cell: $c (see grid/manifest.json)" >&2; exit 2; }
  done
  mapfile -t SELECTED < <(select_cells | grep -xF -f <(printf '%s\n' "${CELL_ARGS[@]}") || true)
else
  mapfile -t SELECTED < <(select_cells)
fi
(( ${#SELECTED[@]} )) || { echo "no cells selected for $ENGINE (subset=$SUBSET)" >&2; exit 2; }

# ---- chunking: a pure function of (SELECTED, JOBS) -------------------------
# Rank r (0-based, in SELECTED's order) goes to part r % JOBS. SELECTED is sorted
# largest-first, so a round robin gives worker 0 the largest cell, worker 1 the
# second largest, and so on, and every worker's own list walks the size ladder
# downward -- the longest cells start first on every core, which is what makes a
# memory or runtime problem show in the first minutes rather than the last.
#
# PURE, and that is the whole point: the same manifest and the same --jobs
# reproduce the identical assignment, so a resumed run looks for a finished cell
# in the part file it was written to. Change the manifest or --jobs between a run
# and its resume and the assignment moves -- run_meta records both so that is
# detectable after the fact.
part_cells() {   # $1 = part index
  local i=0 c
  for c in "${SELECTED[@]}"; do
    (( i % JOBS == $1 )) && printf '%s\n' "$c"
    i=$((i + 1))
  done
}

# ---- engine version, resolved BEFORE the run (it is in the directory name) --
# Mirrors the engine list of the ENGINE_MANIFEST_NAME case above.
case "$ENGINE" in
  lme4|glmmtmb|glmmadaptive)
    PKG=$(case "$ENGINE" in lme4) echo lme4;; glmmtmb) echo glmmTMB;; glmmadaptive) echo GLMMadaptive;; esac)
    ENGINE_VERSION="$(Rscript -e "cat(as.character(packageVersion('$PKG')))")" ;;
  mixedmodels)
    ENGINE_VERSION="$(julia --project="$GRID" -e 'using MixedModels; print(pkgversion(MixedModels))')" ;;
  # The plain crate version, no build-provenance suffix: the engines stamp each
  # record with the version their own build reports, and a suffix added here
  # would put two spellings of one version in one results.jsonl. run_meta's
  # glmm_git_rev is the commit the run started from; run_meta does not record
  # whether the tree was dirty.
  glmm|glmm_wasm) ENGINE_VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/Cargo.toml" | head -1)" ;;
  glmm_python) ENGINE_VERSION="$("$ROOT/python/venv/bin/python" -c 'from importlib.metadata import version; print(version("glmm"))')" ;;
  glmm_r)      ENGINE_VERSION="$(Rscript -e 'cat(as.character(packageVersion("fastglmm")))')" ;;
esac

# ---- run directory: never overwritten --------------------------------------
slug() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\+/-/g; s/^-//; s/-$//'; }
# GRID_MACHINE names the machine in run_meta and the directory name; the hostname
# is the fallback. Keep it fixed per machine: summarize_timing.R refuses to mix
# seconds from runs whose names differ.
MACHINE="$(slug "${GRID_MACHINE:-$(uname -n)}")"
BASE="$(date +%F)_${ENGINE_VERSION}_${MACHINE}"
[[ -n "$LABEL" ]] && BASE="${BASE}_$(slug "$LABEL")"
RUN_ROOT="$GRID/runs/$ENGINE"
# A glmm / port run is committed only when started with --keep; otherwise it lands
# in scratch/, which .gitignore drops. Oracle runs are always committed.
case "$ENGINE" in
  glmm|glmm_wasm|glmm_python|glmm_r) [[ "$KEEP" == 1 ]] || RUN_ROOT="$RUN_ROOT/scratch" ;;
esac
RUN_DIR="$RUN_ROOT/$BASE"
# Rerunning the same engine on the same day appends a counter rather than
# overwriting -- a run directory is an immutable record.
n=2
while [[ -e "$RUN_DIR" ]]; do RUN_DIR="$RUN_ROOT/${BASE}_$n"; n=$((n + 1)); done
mkdir -p "$RUN_DIR"

# Every value here is RECORDED, never set -- no_turbo especially: clock locking
# is the user's bench-l/bench-u, and this script only writes down what it found.
write_run_meta() {   # $1 = the cell_list JSON array
  local cell_list="$1" no_turbo loadavg rev crate r_ver jl_ver
  no_turbo="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
  # The 1/5/15-minute load averages as the run starts: foreign load is invisible
  # to no_turbo and to the pin, and it is what makes two locked runs disagree.
  loadavg="$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo '?')"
  rev="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  crate="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/Cargo.toml" | head -1)"
  r_ver="$(Rscript -e 'cat(as.character(getRversion()))' 2>/dev/null || echo '?')"
  jl_ver="$(julia --version 2>/dev/null | awk '{print $3}' || echo '?')"
  jq -n \
    --arg engine "$ENGINE" --arg engine_version "$ENGINE_VERSION" \
    --arg machine "$MACHINE" --arg date "$(date -Is)" \
    --arg crate "$crate" --arg rev "$rev" --arg subset "$SUBSET" \
    --arg no_turbo "$no_turbo" --arg pin "$PIN_CORES" \
    --arg loadavg "$loadavg" --arg r "$r_ver" --arg jl "$jl_ver" \
    --arg label "$LABEL" \
    --argjson timed "${TIMED_N:-null}" \
    --argjson unlocked "$([[ "$TIMED_UNLOCKED" == 1 ]] && echo true || echo false)" \
    --argjson jobs "$JOBS" --argjson cells "${#SELECTED[@]}" \
    --argjson cell_list "$cell_list" \
    '{engine: $engine, engine_version: $engine_version, machine: $machine, date: $date,
      glmm_crate_version: $crate, glmm_git_rev: $rev, subset: $subset,
      timed: $timed, timed_unlocked: $unlocked, no_turbo: $no_turbo,
      pin_cores: $pin, jobs: $jobs, loadavg_start: $loadavg,
      r_version: $r, julia_version: $jl, label: $label, cells: $cells,
      cell_list: $cell_list}' \
    > "$RUN_DIR/run_meta.json"
  [[ -z "$TIMED_N" || "$no_turbo" == "1" ]] || echo \
    "   WARNING: clock NOT locked (no_turbo=$no_turbo) -- this timed run is marked unlocked" >&2
}
write_run_meta "$CELL_LIST_JSON"

# ---- engine dispatch -------------------------------------------------------
# Exactly the three run-wide variables of the engine environment contract, plus
# per-worker GRID_OUT and GRID_CELLS below -- and no others. The version pins are
# not passed either (each engine reads versions.json out of the grid directory),
# and neither is the watchdog budget: no engine caps its own fits, killing the
# process from outside is the only cell cap.
export GRID_DIR="$GRID" GRID_MANIFEST="$GRID/manifest.json" \
       GRID_TIMED="${TIMED_N:-}"
# Thread counts: every engine fits SERIAL so six workers do not oversubscribe the
# machine. Exported here rather than set in seven scripts. RAYON_NUM_THREADS is
# in the list because faer's default features pull rayon in.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
       RAYON_NUM_THREADS=1
# Mirrors the engine list of the ENGINE_MANIFEST_NAME case above.
case "$ENGINE" in
  glmm)         CMD=(cargo run --quiet --release --manifest-path "$ROOT/Cargo.toml" -p validation --example grid_glmm) ;;
  # WASI passes no host environment unless named: forward the engine contract's
  # variables (GRID_DIR, GRID_MANIFEST, GRID_TIMED, the worker's GRID_OUT and
  # GRID_CELLS) and the thread count.
  glmm_wasm)    CMD=(wasmtime run --dir /::/ --env GRID_DIR --env GRID_MANIFEST --env GRID_TIMED
                     --env GRID_OUT --env GRID_CELLS --env RAYON_NUM_THREADS
                     "$ROOT/target/wasm32-wasip1/release/examples/grid_glmm.wasm") ;;
  glmm_python)  CMD=("$ROOT/python/venv/bin/python" "$GRID/engines/glmm_python.py") ;;
  glmm_r)       CMD=(Rscript "$GRID/engines/glmm_r.R") ;;
  lme4)         CMD=(Rscript "$GRID/engines/lme4.R") ;;
  glmmtmb)      CMD=(Rscript "$GRID/engines/glmmtmb.R") ;;
  glmmadaptive) CMD=(Rscript "$GRID/engines/glmmadaptive.R") ;;
  mixedmodels)  CMD=(julia --project="$GRID" "$GRID/engines/mixedmodels.jl") ;;
esac
# Compile OUTSIDE the watchdog: a first build takes minutes with no output
# writes, which the mtime test would read as a hung cell and kill the compiler.
[[ "$ENGINE" == glmm ]] && cargo build --quiet --release \
  --manifest-path "$ROOT/Cargo.toml" -p validation --example grid_glmm
[[ "$ENGINE" == glmm_wasm ]] && RUSTFLAGS="-C target-feature=+simd128" cargo build --quiet \
  --release --target wasm32-wasip1 --manifest-path "$ROOT/Cargo.toml" -p validation \
  --example grid_glmm

# Per-launch startup grace: loading the engine writes nothing (a Julia package
# load plus the first fit's JIT can exceed a whole cell budget), so until this
# launch appends its first line, judge staleness against the grace instead of
# the budget. GRID_STARTUP_GRACE overrides it; setting it to 0 alongside a tiny
# GRID_WATCHDOG is how the kill-and-record path is exercised on a fast cell.
GRACE="${GRID_STARTUP_GRACE:-$((BUDGET + 180))}"

# The first cell of this worker with no record yet, or failure when the worker is
# done. Reads `out` and `cells` from run_worker, its only caller, through bash's
# dynamic scoping.
next_missing() {
  local done
  # -R + fromjson?: kill -9 can truncate the final line -- skip it rather than
  # fail the whole scan, which would re-flag every finished cell as missing.
  done=$(jq -rR 'fromjson? | .cell' "$out" 2>/dev/null | sort -u) || done=""
  local c
  for c in "${cells[@]}"; do
    grep -qxF "$c" <<< "$done" || { echo "$c"; return 0; }
  done
  return 1
}

done_count() { jq -rR 'fromjson? | .cell' "$out" 2>/dev/null | sort -u | wc -l; }

# One worker: its own part file, its own slice of the cell list, its own
# watchdog. Run as a background subshell, so the exports below are this worker's.
run_worker() {   # $1 = part index
  local part="$1" out MISSING ENGPID LINES0 KILLED EFF NOW MT RC CELL LAST_STUCK=""
  local -a cells pin=()
  mapfile -t cells < <(part_cells "$part")
  (( ${#cells[@]} )) || return 0
  out="$RUN_DIR/results.part$part.jsonl"
  export GRID_OUT="$out" GRID_CELLS="$(printf '%s\n' "${cells[@]}" | paste -sd,)"
  [[ "$PIN_CORES" == none ]] || pin=(taskset -c "$part")
  touch "$out"

  while MISSING=$(next_missing); do
    echo ">> $ENGINE part $part: next cell $MISSING ($(done_count)/${#cells[@]} done)"
    "${pin[@]}" "${CMD[@]}" &
    ENGPID=$!
    LINES0=$(wc -l < "$out")
    KILLED=0
    while kill -0 "$ENGPID" 2>/dev/null; do
      sleep 1
      if (( $(wc -l < "$out") == LINES0 )); then EFF=$GRACE; else EFF=$BUDGET; fi
      NOW=$(date +%s); MT=$(stat -c %Y "$out")
      if (( NOW - MT > EFF )); then
        echo ">> part $part: timeout on $(next_missing || echo '?') -- killing $ENGINE" >&2
        kill -9 "$ENGPID" 2>/dev/null || true
        wait "$ENGPID" 2>/dev/null || true
        CELL=$(next_missing || true)
        if [[ -n "${CELL:-}" ]]; then
          # EFF, not the budget: a launch killed during the startup grace waited
          # that long, and the message has to say what was actually spent.
          printf '{"cell":"%s","engine":"%s","engine_version":"%s","converged":false,"singular":false,"status":"timeout","message":"watchdog killed the engine after %ss","coef_names":[],"beta":[],"varcomp":[],"sigma":null,"nb_theta":null,"loglik":null,"deviance":null,"n_eval":null,"wall_seconds":null,"fits_per_sample":1}\n' \
            "$CELL" "$ENGINE" "$ENGINE_VERSION" "$EFF" >> "$out"
        fi
        KILLED=1
        break
      fi
    done
    RC=0; wait "$ENGPID" 2>/dev/null || RC=$?
    # The engine exited on its own with cells remaining. Its exit code decides
    # what that means, and conflating the two writes fabricated failures for
    # cells nothing tried: in campaigns/speed-campaign 24 consecutive cells, one of
    # them a 4 ms 300-row fit, came out engine-fail with n_eval=0 because the
    # launches never ran.
    #   RC != 0 -- the LAUNCH failed before reaching the cell: build error, OOM
    #     kill, panic at startup. Nothing whatever is known about the cell, so
    #     writing a result for it fabricates data. Two strikes, then stop.
    #   RC == 0 -- the engine ran and declined the cell without writing it. That
    #     IS a cell-level failure and is recorded as one, so the loop can't spin.
    # Skipped after a watchdog kill: the timeout record is already written, and
    # next_missing now names the FOLLOWING cell, which nothing has tried yet.
    if [[ "$KILLED" == 0 ]] && CELL=$(next_missing); then
      if [[ "$CELL" != "$LAST_STUCK" ]]; then
        LAST_STUCK="$CELL"
      elif [[ "$RC" != 0 ]]; then
        echo ">> $ENGINE part $part: launch exited $RC twice with no output on $CELL -- aborting" >&2
        echo ">> (build failure or OOM; nothing is recorded for this cell)" >&2
        return 1
      else
        printf '{"cell":"%s","engine":"%s","engine_version":"%s","converged":false,"singular":false,"status":"engine-fail","message":"launch declined the cell","coef_names":[],"beta":[],"varcomp":[],"sigma":null,"nb_theta":null,"loglik":null,"deviance":null,"n_eval":null,"wall_seconds":null,"fits_per_sample":1}\n' \
          "$CELL" "$ENGINE" "$ENGINE_VERSION" >> "$out"
        LAST_STUCK=""
      fi
    fi
  done
}

PIDS=()
for ((i = 0; i < JOBS; i++)); do
  run_worker "$i" &
  PIDS+=("$!")
done
FAILED=0
for p in "${PIDS[@]}"; do wait "$p" || FAILED=1; done

cat "$RUN_DIR"/results.part*.jsonl > "$RUN_DIR/results.jsonl"
# The parts are KEPT, not deleted: they are this run's own scratch inside its own
# never-overwritten directory, they cost a copy of a few MB, and they are what
# shows which worker fitted which cell when a run has to be picked apart.
echo ">> $ENGINE: ${#SELECTED[@]} cells -> $RUN_DIR/results.jsonl"
(( FAILED == 0 )) || { echo ">> $ENGINE: at least one worker aborted -- results.jsonl is incomplete" >&2; exit 1; }
