#!/usr/bin/env bash
# Moves a finished grid run into its engine's kept slot, the only run folders git
# tracks:
#
#   ./promote.sh <run dir>
#
#   runs/<engine>/last/        the newest full, untimed run -- the accuracy reference
#   runs/<engine>/last-timed/  the newest timed run, plus the cells it did not fit
#
# run.sh calls this at the end of every oracle run and every --keep run, once all
# workers finished, so a run that dies never replaces the run before it. A fast or
# cell-restricted untimed run has no slot and stays where it is.
#
# Replacing a slot deletes nothing: the run it held moves to runs/<engine>/scratch/
# under its own name (`run_name` in its run_meta.json), and git history keeps every
# committed version -- `git log -p -- runs/glmm/last/` is the run history.
#
# TIMED MERGE. An oracle timed run is often a few named cells (the ones added to
# the manifest), so a new timed run replaces only the cells it fitted: the slot's
# records for every other manifest cell are carried into it, and its run_meta.json
# lists the runs they came from under `carried`. summarize_timing.R reads that
# list for provenance and for its one-machine check. It can name a run whose cells
# were all refitted since; it never misses one.
set -euo pipefail
GRID="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(cd "${1:?usage: ./promote.sh <run dir>}" && pwd)"
[[ -f "$RUN/run_meta.json" && -f "$RUN/results.jsonl" ]] \
  || { echo "promote.sh: $RUN is not a run directory" >&2; exit 2; }

read -r ENGINE SUBSET TIMED < <(jq -r '[.engine, .subset, (.timed // "null")] | @tsv' "$RUN/run_meta.json")
if   [[ "$TIMED" != null ]]; then SLOT=last-timed
elif [[ "$SUBSET" == full ]]; then SLOT=last
else
  echo ">> $ENGINE: a $SUBSET untimed run has no kept slot -- it stays in $RUN"
  exit 0
fi
ROOT="$GRID/runs/$ENGINE"
DEST="$ROOT/$SLOT"
[[ "$RUN" != "$DEST" ]] || { echo "promote.sh: $RUN already is the $SLOT slot" >&2; exit 2; }

if [[ -d "$DEST" ]]; then
  if [[ "$SLOT" == last-timed ]]; then
    have="$(jq -rR 'fromjson? | .cell' "$RUN/results.jsonl" | jq -R . | jq -sc .)"
    known="$(jq -c '[.cells[].cell]' "$GRID/manifest.json")"
    carried="$(jq -cR --argjson have "$have" --argjson known "$known" \
      'fromjson? | select(.cell as $c | ($have | index($c) | not) and ($known | index($c)))' \
      "$DEST/results.jsonl")"
    if [[ -n "$carried" ]]; then
      printf '%s\n' "$carried" >> "$RUN/results.jsonl"
      jq --slurpfile old "$DEST/run_meta.json" \
        '.carried = [$old[0] | del(.carried)] + ($old[0].carried // [])' \
        "$RUN/run_meta.json" > "$RUN/run_meta.json.tmp"
      mv "$RUN/run_meta.json.tmp" "$RUN/run_meta.json"
      echo ">> $ENGINE: carried $(printf '%s\n' "$carried" | wc -l) cell(s) from the previous $SLOT run"
    fi
  fi
  # A run that sat in the slot before run_name existed has none; its date stands in.
  old_name="$(jq -r '.run_name // ("unnamed_" + .date)' "$DEST/run_meta.json")"
  mkdir -p "$ROOT/scratch"
  back="$ROOT/scratch/$old_name"
  n=2
  while [[ -e "$back" ]]; do back="$ROOT/scratch/${old_name}_$n"; n=$((n + 1)); done
  mv "$DEST" "$back"
  echo ">> $ENGINE: previous $SLOT run moved to $back"
fi

jq --arg name "$(basename "$RUN")" '.run_name //= $name' "$RUN/run_meta.json" > "$RUN/run_meta.json.tmp"
mv "$RUN/run_meta.json.tmp" "$RUN/run_meta.json"
mv "$RUN" "$DEST"
echo ">> $ENGINE: run is now $DEST"
