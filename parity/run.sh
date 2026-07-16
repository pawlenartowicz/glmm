#!/usr/bin/env bash
# Run the fit engines over manifest datasets, then the agreement check.
# Adding Rust later is ONE list entry (`rust`) plus oracle/fit.rs -- no restructuring.
#
# Shared by this suite and parity/weights/run.sh (the FitOptions.weights suite),
# which execs this script after exporting PARITY_SUITE_DIR and the
# PARITY_PREP_SCRIPT / PARITY_PREP_DESC / PARITY_RESULTS_DESC / PARITY_README_HINT /
# PARITY_MANIFEST_HINT overrides -- see weights/run.sh for the exact values. Any
# suite-specific text belongs behind one of those vars, not hardcoded here.
#
#   ./run.sh                 fit glmm (Rust) AND the glmm Python port + compare against
#                             the EXISTING results/lme4_*/mixedmodels_* JSONs on disk
#                             (fast default; R/lme4 and Julia/MixedModels are NOT refit --
#                             those results don't change run to run, so paying their cost
#                             every time is wasted). Never touches data_{empirical,simulated}/.
#                             The port fits the same kernel through PyO3: compare.R gates it
#                             against the Rust row (near-exact), and summarize_timing.R's
#                             py_gap column is the end-to-end cost of calling from Python.
#   ./run.sh --oracles       refit ALL FOUR engines (R, Julia, Rust, Python) + compare -- use when
#                             regenerating the oracle itself (new dataset, new machine,
#                             tolerance work).
#   ./run.sh --prep          regenerate the committed data_{empirical,simulated}/*.csv
#                             first. IMPLIES --oracles: changed data invalidates the old
#                             R/Julia results, so a --prep run always refits all three
#                             (silently upgrading --prep alone to also refit lme4/MM would
#                             be surprising; --prep --oracles is accepted as the same thing).
#   ./run.sh [flags] ds...   restrict every engine that runs to the named manifest
#                             datasets (e.g. `./run.sh cbpp`, `./run.sh --oracles cbpp
#                             sleepstudy`) -- validated against manifest.json, unknown
#                             names fail loudly before anything is fit.
set -euo pipefail
PARITY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Suite directory (manifest.json, data_*/, results/) -- defaults to this script's
# own directory (the main suite). weights/run.sh exports PARITY_SUITE_DIR to its
# own dir before exec'ing here, so oracle/fit.{R,jl,rs} and compare.R (which read
# the same var, defaulting the same way) resolve the weights suite instead.
: "${PARITY_SUITE_DIR:=$PARITY}"
export PARITY_SUITE_DIR
: "${PARITY_PREP_SCRIPT:=export_data.R}"
: "${PARITY_PREP_DESC:=data_{empirical,simulated}/*.csv from lme4}"
: "${PARITY_RESULTS_DESC:=glmm_{empirical,simulated}/}"
: "${PARITY_README_HINT:=README setup}"
: "${PARITY_MANIFEST_HINT:=parity/manifest.json}"

PREP=0
ORACLES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep) PREP=1; shift ;;
    --oracles) ORACLES=1; shift ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
# --prep implies --oracles: regenerated data invalidates the frozen lme4/MixedModels
# results, so they must be refit alongside Rust (see header comment).
[[ "$PREP" == 1 ]] && ORACLES=1

# Positional args left after flag parsing are manifest dataset names. Validate against
# manifest.json's "name" fields with grep (jq is not assumed installed) so an unknown
# name fails loudly before any engine runs, rather than silently fitting nothing.
if [[ $# -gt 0 ]]; then
  for ds in "$@"; do
    grep -qF "\"name\": \"$ds\"" "$PARITY_SUITE_DIR/manifest.json" \
      || { echo "unknown dataset: $ds (see $PARITY_MANIFEST_HINT)" >&2; exit 2; }
  done
  PARITY_ONLY="$(IFS=,; echo "$*")"
  export PARITY_ONLY
fi

ENGINES=(rust py glmm_r)
[[ "$ORACLES" == 1 ]] && ENGINES=(lme4 jl rust py glmm_r)

# Pin the timed fits to one P-core (cores 0-5 are the 5.3 GHz P-cores on this box;
# same core every run) so a locked-machine run isn't perturbed by the scheduler
# hopping onto a slower E-core (6-13) or LP-E core (14-15). No-op if taskset is
# absent. This does NOT lock the machine -- run `bench-l` yourself first, else the
# numbers are powersave noise regardless of pinning. compare.R is untimed, unpinned.
PIN=""
command -v taskset >/dev/null && PIN="taskset -c 1"

if [[ "$PREP" == 1 ]]; then
  echo ">> prep: regenerating committed $PARITY_PREP_DESC"
  Rscript "$PARITY_SUITE_DIR/prep/$PARITY_PREP_SCRIPT"
fi

for e in "${ENGINES[@]}"; do
  case "$e" in
    lme4)
      echo ">> lme4 (R)"
      $PIN Rscript "$PARITY/oracle/fit.R" ;;
    jl)
      echo ">> MixedModels (Julia)"
      if ! command -v julia >/dev/null || [[ ! -f "$PARITY/Manifest.toml" ]]; then
        echo "   skipped: julia or the pinned env is missing (see $PARITY_README_HINT)" >&2
      else
        $PIN julia --project="$PARITY" "$PARITY/oracle/fit.jl"
      fi ;;
    rust)
      echo ">> glmm (Rust) -> results/$PARITY_RESULTS_DESC"
      if ! command -v cargo >/dev/null; then
        echo "   skipped: cargo not found" >&2
      else
        $PIN cargo run --quiet --release --manifest-path "$PARITY/../Cargo.toml" \
          -p parity --example parity_fit
      fi ;;
    py)
      echo ">> glmm python port -> results/glmm_python_*/"
      # The repo's own venv first (python/venv, where `maturin develop --release`
      # installs the wheel editable), else whatever python3 has glmm importable.
      # The wheel MUST be a --release build: a debug kernel would report the port's
      # "overhead" as a codegen artifact an order of magnitude too large.
      # dirname "$PARITY" (already absolute), not "$PARITY/..": a literal ".." in
      # sys.executable makes the venv's site.py warn about an unexpected sys.prefix.
      PY="$(dirname "$PARITY")/python/venv/bin/python"
      [[ -x "$PY" ]] || PY="$(command -v python3 || true)"
      if [[ -z "$PY" ]] || ! "$PY" -c 'import glmm' 2>/dev/null; then
        echo "   skipped: no python with the glmm wheel installed (see $PARITY_README_HINT)" >&2
      else
        $PIN "$PY" "$PARITY/oracle/fit.py"
      fi ;;
    glmm_r)
      echo ">> glmm R port -> results/glmm_r_*/"
      # Same kernel as the Rust/Python engines, reached through the fastglmm R
      # package (extendr wrapper). No venv step -- the package is installed in the
      # R library. Skip cleanly if it is not, so a machine without it still runs
      # the rest.
      if ! Rscript -e 'if (!requireNamespace("fastglmm", quietly=TRUE)) quit(status=1)' 2>/dev/null; then
        echo "   skipped: fastglmm R package not installed (see $PARITY_README_HINT)" >&2
      else
        $PIN Rscript "$PARITY/oracle/fit_port.R"
      fi ;;
    *)
      echo "unknown engine: $e" >&2; exit 2 ;;
  esac
done

echo ">> compare"
Rscript "$PARITY/compare.R"
