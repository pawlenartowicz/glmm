#!/usr/bin/env bash
# Regenerate the 24 datasets with the paper's own generator (set.seed(126),
# deterministic). Needs only MASS + jsonlite - NOT the estimator packages, since
# we do not run their estimators. Writes external/data/<cell>.RData.
set -euo pipefail
ACC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$ACC/external"
cd "$EXT"
echo "== 00_SetHyperparameters (config_1..24.json) =="
Rscript scripts/00_SetHyperparameters.R
echo "== 01_GenerateData (24 cells x 1000 sims) =="
Rscript scripts/01_GenerateData.R
echo "== done: $(ls data/*.RData | wc -l) datasets in $EXT/data =="
