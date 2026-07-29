#!/usr/bin/env bash
# Peak-RSS collection over the manifest (43 rungs) and the large synthetic
# models (models.json) -- one leg at a time. NOT a validation gate: nothing
# here is compared against an oracle, and no clock lock is required (peak RSS
# does not depend on CPU frequency, unlike run.sh's timed legs) -- that is the
# one real advantage this measurement has over the timing one, so don't block
# a collection run on a bench session.
#
# One process per (engine, dataset): /usr/bin/time reports the max RSS across
# a process AND its waited-for children, so wrapping `cargo run` would measure
# cargo, not the fit -- the Rust engine is therefore built once
# (`cargo build --release`) and invoked as the built binary, never through
# `cargo run`. Every engine fits the whole manifest in one process today, so
# this is the only way to get a per-dataset number out of any of them.
#
#   ./memory.sh <leg> --manifest [ds...]        [--engines e1,e2,...]
#   ./memory.sh <leg> --large [id...]           [--engines e1,e2,...]
#   ./memory.sh --oracles                       (large-model rows 1,4,6,9 only,
#                                                 lme4 + MixedModels, written once
#                                                 to results/memory/oracles.tsv --
#                                                 these two engines don't change
#                                                 between legs, so they are never
#                                                 refit per leg)
#   ./memory.sh baselines                       (one load-only process per engine,
#                                                 written once to
#                                                 results/memory/baselines.tsv --
#                                                 see below)
#
# Output: results/memory/<leg>.tsv (or oracles.tsv), columns
#   engine  dataset  rung  n  levels  peak_rss_kb
# plus a run_meta.json per leg (mirrors results/glmm_parallel/run_meta.json).
# A leg's TSV is appended to, not overwritten, across separate --manifest and
# --large invocations targeting the same <leg> name -- so a full leg (both
# dataset classes) is collected as two calls into one file.
#
# Baselines (results/memory/baselines.tsv, columns engine baseline_kb): peak
# RSS of the same engine with NO fit performed, i.e. process start-up alone
# (binary exec / interpreter boot / package load). summarize_memory.R's
# cross-engine view subtracts this from a large model's peak RSS so that view
# reads as fit cost, not runtime footprint. One process per engine, same
# /usr/bin/time -f '%M' mechanism as every other leg:
#   rust         -- memory_fit --noop (loads the binary, does nothing else)
#   glmm_python  -- python -c 'import glmm' (no fit)
#   glmm_r       -- Rscript -e 'library(fastglmm)' (no fit)
#   lme4         -- Rscript -e 'library(lme4)'
#   mixedmodels  -- julia --project=. -e 'using MixedModels, CSV, DataFrames'
#                   (slow start, ~60-120s -- that's expected, not a hang)
# This is a one-off collection, not appended per leg like the other TSVs --
# rerunning `baselines` overwrites the file (the numbers are process start-up
# cost, which doesn't change across glmm code revisions the way a fit does).
#
# `levels` is an approximation of k_total (src/glmm/workspace.rs's dense
# allocation term): for a manifest rung it is the summed distinct-value count
# over manifest.json's "factors" list for that dataset (exact only for
# intercept-only groupings -- a random SLOPE multiplies a grouping's k by q,
# which a plain distinct-value count can't see). For a large model it is
# read directly from models.json's k_total_predicted, which IS exact (the
# generator controls level counts and slope columns by construction).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/memory"
RESULTS="$ROOT/results/memory"
mkdir -p "$RESULTS" "$RESULTS/data"

ENGINES="rust,py,glmm_r"
MODE=""
LEG=""
SUBSET_ARGS=()

if [[ "${1:-}" == "--oracles" ]]; then
  MODE="oracles"
  shift
elif [[ "${1:-}" == "baselines" ]]; then
  MODE="baselines"
  shift
else
  LEG="${1:?usage: memory.sh <leg> --manifest|--large [ds/id...] [--engines e1,e2,...], or memory.sh --oracles}"
  shift
  case "${1:-}" in
    --manifest) MODE="manifest"; shift ;;
    --large) MODE="large"; shift ;;
    *) echo "expected --manifest or --large after the leg name" >&2; exit 2 ;;
  esac
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --engines) ENGINES="$2"; shift 2 ;;
      -*) echo "unknown flag: $1" >&2; exit 2 ;;
      *) SUBSET_ARGS+=("$1"); shift ;;
    esac
  done
fi

# Build once, release, and invoke the binary directly (never `cargo run`) --
# see header. Shared by every engine selection: the "rust" engine below reads
# it, and it costs nothing extra when only py/glmm_r are selected.
RUST_BIN="$ROOT/../target/release/examples/memory_fit"
build_rust() {
  [[ -x "$RUST_BIN" ]] && return
  cargo build --quiet --release --manifest-path "$ROOT/../Cargo.toml" \
    -p validation --example memory_fit
}

PY="$(dirname "$ROOT")/python/venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"

# n/levels for a manifest CSV: n = data row count, levels = summed distinct-
# value count over the given factor columns (an approximation of k_total --
# see header). Single awk pass, no jq/python dependency (mirrors run.sh's
# "jq not assumed installed" stance).
csv_stats() {
  local csv="$1" factors_csv="$2"
  awk -v factors="$factors_csv" -F',' '
    NR == 1 {
      n_factors = split(factors, want, ",")
      # Header fields carry the corpus quoting ("herd","period",...) but the
      # factor names passed in never do (manifest.json bare strings) -- strip
      # quotes before indexing or every lookup misses and fidx[k] stays 0,
      # which awk reads as $0 (the whole line) instead of the intended column.
      for (j = 1; j <= NF; j++) { h = $j; gsub(/^"|"$/, "", h); idx[h] = j }
      for (k = 1; k <= n_factors; k++) fidx[k] = idx[want[k]]
      next
    }
    NF == 0 { next }
    {
      n++
      for (k = 1; k <= n_factors; k++) seen[k, $(fidx[k])] = 1
    }
    END {
      # `seen` is keyed "<column index>SUBSEP<value>" (one entry per distinct
      # value actually observed for that column) -- levels is the number of
      # distinct keys per column, summed over the factor columns.
      levels = 0
      for (k = 1; k <= n_factors; k++) {
        cnt = 0
        for (key in seen) {
          split(key, parts, SUBSEP)
          if (parts[1] == k) cnt++
        }
        levels += cnt
      }
      print n, levels
    }
  ' "$csv"
}

manifest_factors() {
  python3 -c "
import json, sys
m = json.load(open('$ROOT/manifest.json'))
for d in m['datasets']:
    if d['name'] == sys.argv[1]:
        print(','.join(d.get('factors', [])))
        print(d['rung'])
        print('sim' if d.get('source') == 'sim' else 'not-sim')
        # Some manifest rows share a CSV with another row under a different
        # analysis (e.g. cbpp_probit reuses cbpp.csv via 'data': 'cbpp') --
        # the file on disk is named after 'data', not the dataset's own
        # 'name'. Emit it so the caller resolves the real path instead of
        # assuming name == csv basename.
        print(d.get('data', d['name']))
        sys.exit(0)
sys.exit('dataset not found: ' + sys.argv[1])
" "$1"
}

manifest_names() {
  python3 -c "
import json
m = json.load(open('$ROOT/manifest.json'))
print('\n'.join(d['name'] for d in m['datasets']))
"
}

large_model_json() {
  python3 -c "
import json, sys
m = json.load(open('$HERE/models.json'))
for row in m['models']:
    if row['id'] == int(sys.argv[1]):
        print(json.dumps(row))
        sys.exit(0)
sys.exit('no model id ' + sys.argv[1])
" "$1"
}

# Append one measured row to the leg's TSV, writing the header first if the
# file is new.
emit() {
  local out="$1" engine="$2" dataset="$3" rung="$4" n="$5" levels="$6" kb="$7"
  if [[ ! -f "$out" ]]; then
    printf 'engine\tdataset\trung\tn\tlevels\tpeak_rss_kb\n' > "$out"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$engine" "$dataset" "$rung" "$n" "$levels" "$kb" >> "$out"
}

# Peak RSS in KB via /usr/bin/time -f '%M' -- the same max-RSS source
# toy28/ram.sh reads (there via -v's "Maximum resident set size"); -f '%M'
# reports just the number, which is all this harness needs.
#
# Two separate files, not one: GNU time's -o file is opened in truncate mode
# once the child exits and its own report is written into it, so a single
# file fed by both "child stderr appended during the run" and "-o's report
# written after" loses the child's stderr -- on a real failure you'd see only
# time's "Command exited with non-zero status N" and nothing about why.
measure() {
  local errfile timefile
  errfile="$(mktemp)"
  timefile="$(mktemp)"
  /usr/bin/time -f '%M' -o "$timefile" "$@" >/dev/null 2>"$errfile" || {
    echo "   FIT FAILED: $*" >&2
    echo "   --- child stderr ---" >&2
    cat "$errfile" >&2
    echo "   --- /usr/bin/time report ---" >&2
    cat "$timefile" >&2
    trash-put "$errfile" "$timefile" 2>/dev/null || rm -f "$errfile" "$timefile"
    return 1
  }
  tail -n1 "$timefile"
  trash-put "$errfile" "$timefile" 2>/dev/null || true
}

run_manifest_leg() {
  local out="$RESULTS/$LEG.tsv"
  local validation_bin="$ROOT/../target/release/examples/validation_fit"
  local names
  if [[ ${#SUBSET_ARGS[@]} -gt 0 ]]; then
    for ds in "${SUBSET_ARGS[@]}"; do
      grep -qF "\"name\": \"$ds\"" "$ROOT/manifest.json" \
        || { echo "unknown dataset: $ds (see validation/manifest.json)" >&2; exit 2; }
    done
    names=("${SUBSET_ARGS[@]}")
  else
    mapfile -t names < <(manifest_names)
  fi

  IFS=',' read -ra engine_list <<< "$ENGINES"
  for ds in "${names[@]}"; do
    # Read the four manifest_factors lines positionally (mapfile), not via
    # `tr '\n' ' ' | read` -- five manifest datasets have "factors": [] (rungs
    # 29,30,39,40,42), which prints an empty first line; flattening newlines
    # to spaces before splitting on whitespace then shifts every field left
    # (rung ends up in $factors, "sim"/"not-sim" ends up in $rung, and $is_sim
    # goes empty), silently routing an empirical-looking csv path and
    # corrupting rung/n/levels. mapfile keeps each line in its own slot
    # regardless of whether it's empty.
    mapfile -t _mf < <(manifest_factors "$ds")
    local factors="${_mf[0]}" rung="${_mf[1]}" is_sim="${_mf[2]}" csv_name="${_mf[3]}"
    local source_dir="empirical"; [[ "$is_sim" == "sim" ]] && source_dir="simulated"
    # csv_name is manifest's "data" field (falls back to $ds when absent) --
    # rows like cbpp_probit reuse another row's CSV under a different
    # analysis, so the file on disk is not always "$ds.csv".
    local csv="$ROOT/data/$source_dir/$csv_name.csv"
    # csv_stats("", ...): awk's split("", want, ",") yields n_factors=0, so
    # levels=0 for a no-factor dataset -- correct, not a fallback: there is no
    # grouping term to size, so k_total's grouping contribution really is 0.
    read -r n levels <<< "$(csv_stats "$csv" "$factors")"

    for e in "${engine_list[@]}"; do
      case "$e" in
        rust)
          build_rust
          [[ -x "$validation_bin" ]] || cargo build --quiet --release \
            --manifest-path "$ROOT/../Cargo.toml" -p validation --example validation_fit
          kb="$(VALIDATION_ONLY="$ds" measure "$validation_bin")" \
            && emit "$out" rust "$ds" "$rung" "$n" "$levels" "$kb"
          ;;
        py)
          if [[ -x "$PY" ]] && "$PY" -c 'import glmm' 2>/dev/null; then
            kb="$(VALIDATION_ONLY="$ds" measure "$PY" "$ROOT/engines/glmm_python.py")" \
              && emit "$out" py "$ds" "$rung" "$n" "$levels" "$kb"
          else
            echo "   skipped py: no python with the glmm wheel installed" >&2
          fi
          ;;
        glmm_r)
          if Rscript -e 'if (!requireNamespace("fastglmm", quietly=TRUE)) quit(status=1)' 2>/dev/null; then
            kb="$(VALIDATION_ONLY="$ds" measure Rscript "$ROOT/engines/glmm_r.R")" \
              && emit "$out" glmm_r "$ds" "$rung" "$n" "$levels" "$kb"
          else
            echo "   skipped glmm_r: fastglmm not installed" >&2
          fi
          ;;
        *) echo "unknown engine: $e (manifest legs support rust,py,glmm_r)" >&2; exit 2 ;;
      esac
    done
  done
  write_run_meta "$out"
}

run_large_leg() {
  local out="$RESULTS/$LEG.tsv"
  local ids=("${SUBSET_ARGS[@]}")
  if [[ ${#ids[@]} -eq 0 ]]; then ids=($(seq 1 13)); fi

  IFS=',' read -ra engine_list <<< "$ENGINES"
  for id in "${ids[@]}"; do
    local row; row="$(large_model_json "$id")"
    local n formula family link factors levels nagq
    n="$(echo "$row" | python3 -c 'import json,sys; print(json.load(sys.stdin)["n"])')"
    formula="$(echo "$row" | python3 -c 'import json,sys; print(json.load(sys.stdin)["formula"])')"
    family="$(echo "$row" | python3 -c 'import json,sys; print(json.load(sys.stdin)["family"])')"
    link="$(echo "$row" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("link",""))')"
    factors="$(echo "$row" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(",".join(g["name"] for g in d["groups"]))')"
    levels="$(echo "$row" | python3 -c 'import json,sys; d=json.load(sys.stdin); k=d["k_total_predicted"]; print(k if k is not None else "NA")')"
    nagq="$(echo "$row" | python3 -c 'import json,sys; print(json.load(sys.stdin)["nagq"])')"

    local csv="$RESULTS/data/model_$id.csv"
    [[ -f "$csv" ]] || "$PY" "$HERE/gen_models.py" "$id" "$csv" >/dev/null

    for e in "${engine_list[@]}"; do
      case "$e" in
        rust)
          build_rust
          kb="$(measure "$RUST_BIN" "$csv" "$formula" "$family" "$link" "$factors" "$nagq")" \
            && emit "$out" rust "large_$id" "L$id" "$n" "$levels" "$kb"
          ;;
        py)
          if [[ -x "$PY" ]] && "$PY" -c 'import glmm' 2>/dev/null; then
            kb="$(measure "$PY" "$HERE/fit_python.py" "$csv" "$formula" "$family" "$link" "$factors" "$nagq")" \
              && emit "$out" py "large_$id" "L$id" "$n" "$levels" "$kb"
          else
            echo "   skipped py: no python with the glmm wheel installed" >&2
          fi
          ;;
        glmm_r)
          if Rscript -e 'if (!requireNamespace("fastglmm", quietly=TRUE)) quit(status=1)' 2>/dev/null; then
            kb="$(measure Rscript "$HERE/fit_r.R" "$csv" "$formula" "$family" "$link" "$factors" "$nagq")" \
              && emit "$out" glmm_r "large_$id" "L$id" "$n" "$levels" "$kb"
          else
            echo "   skipped glmm_r: fastglmm not installed" >&2
          fi
          ;;
        *) echo "unknown engine: $e (large legs support rust,py,glmm_r -- lme4/mmjl are --oracles only)" >&2; exit 2 ;;
      esac
    done
  done
  write_run_meta "$out"
}

run_oracles() {
  local out="$RESULTS/oracles.tsv"
  for id in 1 4 6 9; do
    local row; row="$(large_model_json "$id")"
    local n formula factors levels
    n="$(echo "$row" | python3 -c 'import json,sys; print(json.load(sys.stdin)["n"])')"
    formula="$(echo "$row" | python3 -c 'import json,sys; print(json.load(sys.stdin)["formula"])')"
    factors="$(echo "$row" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(",".join(g["name"] for g in d["groups"]))')"
    levels="$(echo "$row" | python3 -c 'import json,sys; print(json.load(sys.stdin)["k_total_predicted"])')"

    local csv="$RESULTS/data/model_$id.csv"
    [[ -f "$csv" ]] || "$PY" "$HERE/gen_models.py" "$id" "$csv" >/dev/null

    if command -v Rscript >/dev/null; then
      kb="$(measure Rscript "$HERE/fit_lme4.R" "$csv" "$formula" "$factors")" \
        && emit "$out" lme4 "large_$id" "L$id" "$n" "$levels" "$kb"
    else
      echo "   skipped lme4: Rscript not found" >&2
    fi
    if command -v julia >/dev/null; then
      kb="$(measure julia --project="$ROOT" "$HERE/fit_mixedmodels.jl" "$csv" "$formula" "$factors")" \
        && emit "$out" mixedmodels "large_$id" "L$id" "$n" "$levels" "$kb"
    else
      echo "   skipped mixedmodels: julia not found" >&2
    fi
  done
  write_run_meta "$out"
}

# One load-only process per engine -- see header for what each is measuring
# and why. Overwrites baselines.tsv (not appended -- see header).
run_baselines() {
  local out="$RESULTS/baselines.tsv"
  printf 'engine\tbaseline_kb\n' > "$out"
  ENGINES="rust,glmm_python,glmm_r,lme4,mixedmodels" # for write_run_meta's engines_present field

  build_rust
  kb="$(measure "$RUST_BIN" --noop)" \
    && printf 'rust\t%s\n' "$kb" >> "$out"

  if [[ -x "$PY" ]] && "$PY" -c 'import glmm' 2>/dev/null; then
    kb="$(measure "$PY" -c 'import glmm')" \
      && printf 'glmm_python\t%s\n' "$kb" >> "$out"
  else
    echo "   skipped glmm_python: no python with the glmm wheel installed" >&2
  fi

  if Rscript -e 'if (!requireNamespace("fastglmm", quietly=TRUE)) quit(status=1)' 2>/dev/null; then
    kb="$(measure Rscript -e 'suppressMessages(library(fastglmm))')" \
      && printf 'glmm_r\t%s\n' "$kb" >> "$out"
  else
    echo "   skipped glmm_r: fastglmm not installed" >&2
  fi

  if command -v Rscript >/dev/null; then
    kb="$(measure Rscript -e 'suppressMessages(library(lme4))')" \
      && printf 'lme4\t%s\n' "$kb" >> "$out"
  else
    echo "   skipped lme4: Rscript not found" >&2
  fi

  if command -v julia >/dev/null; then
    kb="$(measure julia --project="$ROOT" -e 'using MixedModels, CSV, DataFrames')" \
      && printf 'mixedmodels\t%s\n' "$kb" >> "$out"
  else
    echo "   skipped mixedmodels: julia not found" >&2
  fi

  write_run_meta "$out"
}

write_run_meta() {
  local out_tsv="$1"
  local meta="${out_tsv%.tsv}_run_meta.json"
  local rev; rev="$(git -C "$ROOT/.." rev-parse HEAD 2>/dev/null || echo unknown)"
  python3 -c "
import json, platform, sys
meta = {
    'date': '$(date -I)',
    'glmm_git_rev': '$rev',
    'engines_present': '$ENGINES',
    'machine': platform.platform(),
    'clock_lock': 'not required -- peak RSS does not depend on CPU frequency',
}
json.dump(meta, open('$meta', 'w'), indent=2)
"
}

case "$MODE" in
  manifest) run_manifest_leg ;;
  large) run_large_leg ;;
  oracles) run_oracles ;;
  baselines) run_baselines ;;
esac
