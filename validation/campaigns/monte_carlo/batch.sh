#!/usr/bin/env bash
# One escalating batch: fit all three engines up to a cumulative rep count
# (ACC_UPTO), then summarize. Append-only, so a batch fits only the new reps.
# Pinned to core 1 (locked machine) for clean timing.
#   ./batch.sh 10      # reps 1..10 for every engine, then print
set -euo pipefail
ACC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPTO="${1:?usage: batch.sh <cumulative-rep-count>}"
PIN=""; command -v taskset >/dev/null && PIN="taskset -c 1"

# glmm (the accuracy deliverable) runs the full schedule to 1000. lme4 and
# GLMMadaptive are timing/gate references only - their ACCURACY is the frozen
# oracle - so they cap out early: lme4 at 200 (gate + Laplace timing), the much
# slower GLMMadaptive at 100 (adaptive timing). Beyond the cap the append step
# is a no-op, so later batches only advance glmm.
LME4_CAP=200; GADAPT_CAP=100
min() { echo $(( $1 < $2 ? $1 : $2 )); }

if [[ ! -f "$ACC/external/data/n50_M3_Bernoulli_rdi.RData" ]]; then bash "$ACC/gen.sh"; fi

echo "===== batch -> reps 1..$UPTO : fitting ====="
echo "-- glmm (upto $UPTO) --";                       ACC_UPTO="$UPTO"                 $PIN Rscript "$ACC/fit_glmm.R"
echo "-- lme4 (upto $(min $UPTO $LME4_CAP)) --";       ACC_UPTO="$(min $UPTO $LME4_CAP)"   $PIN Rscript "$ACC/fit_lme4.R"
echo "-- GLMMadaptive (upto $(min $UPTO $GADAPT_CAP)) --"; ACC_UPTO="$(min $UPTO $GADAPT_CAP)" $PIN Rscript "$ACC/fit_glmmadaptive.R"
echo "===== batch -> reps 1..$UPTO : summary ====="
# summarize glmm over the full batch; references over their own (smaller) rep counts.
ACC_UPTO="$UPTO" Rscript "$ACC/summarize_accuracy_truth.R"
