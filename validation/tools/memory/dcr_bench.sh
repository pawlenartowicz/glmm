#!/usr/bin/env bash
# The weight-power and CR2 benchmark: wall time and peak RSS
# of fit_dcr.R's arms on gen_dcr_bench.py's datasets, one process per run,
# from /usr/bin/time. A measurement, not a gate. Run it on a locked clock and a
# quiet machine; the R library in $R_LIBS must hold the fastglmm under test,
# nlme 3.1-168 and clubSandwich 0.7.0.
#
#   ./dcr_bench.sh <out_dir> [dataset...]     default: every dataset, in run order
#
# Per (dataset, arm): one run that is dropped, then three, of which the
# summary takes the minimum; a first run over 60 s is the only run. Each run is
# capped at 30 minutes and 16 GB of address space; a capped run is recorded as
# time-cap or mem-cap and that arm skips the rest of the dataset's row.
# Arm 3c runs only on the G = 100,000 datasets. ARMS (default "0 1 2 3 3c 4")
# limits the arms run. When `pin` is on the PATH the whole script runs under
# one `pin --p 1`, which holds its core's lock from the first run to the last.
#
# Output: <out_dir>/runs.tsv (dataset row arm rep wall_s peak_kb status),
# appended to, and <out_dir>/<dataset>.<arm>.<rep>.log; data in <out_dir>/data.
# summarize_dcr_bench.py turns runs.tsv into the report's tables.
set -euo pipefail
# One lock for the whole run: a lock taken per run frees the core between runs.
if [[ -z ${DCR_PINNED:-} ]] && command -v pin >/dev/null; then
  DCR_PINNED=1 exec pin --p 1 -- "$0" "$@"
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: dcr_bench.sh <out_dir> [dataset...]}"
shift
mkdir -p "$OUT/data"
TSV="$OUT/runs.tsv"
[[ -f $TSV ]] || printf 'dataset\trow\tarm\trep\twall_s\tpeak_kb\tstatus\n' >"$TSV"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
CAP_S=1800
CAP_BYTES=$((16 * 1024 * 1024 * 1024))

# One run: prints "<wall_s> <peak_kb> <status>".
run_once() {
  local csv=$1 formula=$2 arm=$3 log=$4 tf rc=0
  tf=$(mktemp)
  prlimit --as=$CAP_BYTES -- timeout $CAP_S \
    /usr/bin/time -f '%e %M' -o "$tf" Rscript "$HERE/fit_dcr.R" "$arm" "$csv" "$formula" \
    >"$log" 2>&1 || rc=$?
  local status=ok
  if ((rc == 124)); then
    status=time-cap
  elif ((rc != 0)); then
    if grep -qiE 'cannot allocate|memory allocation|std::bad_alloc' "$log"; then status=mem-cap; else status=error; fi
  fi
  # /usr/bin/time writes "Command exited with non-zero status" first on a failure.
  read -r wall peak < <(tail -n 1 "$tf" 2>/dev/null || echo "NA NA")
  [[ $status == time-cap ]] && wall=$CAP_S peak=${peak:-NA}
  rm -f "$tf"
  echo "${wall:-NA} ${peak:-NA} $status"
}

LIST=$("${PYTHON:-python3}" "$HERE/gen_dcr_bench.py" --list)  # fails the script, unlike <(...)
declare -A SKIP
while IFS=$'\t' read -r id row formula; do
  if (($#)) && [[ " $* " != *" $id "* ]]; then continue; fi
  csv="$OUT/data/$id.csv"
  [[ -f $csv ]] || "${PYTHON:-python3}" "$HERE/gen_dcr_bench.py" "$id" "$csv"
  arms=(0 1 2 3 4)
  [[ $id == *G100000* || $id == wide-small-* ]] && arms+=(3c)
  for arm in "${arms[@]}"; do
    [[ " ${ARMS:-0 1 2 3 3c 4} " == *" $arm "* ]] || continue
    if [[ -n ${SKIP[$row.$arm]:-} ]]; then
      printf '%s\t%s\t%s\t0\tNA\tNA\tskipped\n' "$id" "$row" "$arm" >>"$TSV"
      continue
    fi
    for rep in 0 1 2 3; do
      read -r wall peak status < <(run_once "$csv" "$formula" "$arm" "$OUT/$id.$arm.$rep.log")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$row" "$arm" "$rep" "$wall" "$peak" "$status" >>"$TSV"
      echo "$id arm $arm rep $rep: ${wall}s ${peak}kB $status"
      if [[ $status != ok ]]; then
        [[ $status == *-cap ]] && SKIP[$row.$arm]=1
        break
      fi
      if ((rep == 0)) && awk -v w="$wall" 'BEGIN { exit !(w > 60) }'; then break; fi
    done
  done
done <<<"$LIST"
