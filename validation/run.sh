#!/usr/bin/env bash
# Run the fit engines over manifest datasets (all 43 rungs, incl. the prior-
# weights tier 29-43), then the agreement check.
#
#   ./run.sh                 fit glmm (Rust) AND the Python + R ports, compare against
#                             the EXISTING results/lme4_*/mixedmodels_* JSONs on disk
#                             (fast default; R/lme4 and Julia/MixedModels are NOT refit).
#   ./run.sh --oracles       refit ALL engines (R, Julia, Rust, Python, R port) + compare.
#   ./run.sh --prep          regenerate data_{empirical,simulated}/*.csv first (both prep
#                             scripts: export_data.R for rungs 1-28, gen_weights_data.R
#                             for 29-43). IMPLIES --oracles: changed data invalidates the
#                             old R/Julia results.
#   ./run.sh --rust-tier2    ALSO run the crate's own cross-engine tier
#                             (`cargo test -p glmm --features oracle-tests`) first.
#   ./run.sh [flags] ds...   restrict every engine that runs to the named manifest
#                             datasets -- validated against manifest.json, unknown
#                             names fail loudly before anything is fit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREP=0
ORACLES=0
TIER2=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep) PREP=1; shift ;;
    --oracles) ORACLES=1; shift ;;
    --rust-tier2) TIER2=1; shift ;;
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
    grep -qF "\"name\": \"$ds\"" "$ROOT/manifest.json" \
      || { echo "unknown dataset: $ds (see validation/manifest.json)" >&2; exit 2; }
  done
  VALIDATION_ONLY="$(IFS=,; echo "$*")"
  export VALIDATION_ONLY
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

# Before the engines, not after: Tier 2 is the cheaper claim and needs no R or
# Julia, so a crate regression is reported without first paying for a full oracle
# refit. Unpinned -- it is a correctness gate, nothing here is timed.
if [[ "$TIER2" == 1 ]]; then
  echo ">> glmm Tier 2 (cross-engine tests)"
  cargo test --quiet --manifest-path "$ROOT/../Cargo.toml" -p glmm --features oracle-tests
fi

if [[ "$PREP" == 1 ]]; then
  echo ">> prep: regenerating committed data_{empirical,simulated}/*.csv"
  Rscript "$ROOT/prep/export_data.R"
  Rscript "$ROOT/prep/gen_weights_data.R"
fi

for e in "${ENGINES[@]}"; do
  case "$e" in
    lme4)
      echo ">> lme4 (R)"
      $PIN Rscript "$ROOT/engines/lme4.R" ;;
    jl)
      echo ">> MixedModels (Julia)"
      if ! command -v julia >/dev/null || [[ ! -f "$ROOT/Manifest.toml" ]]; then
        echo "   skipped: julia or the pinned env is missing (see README setup)" >&2
      else
        $PIN julia --project="$ROOT" "$ROOT/engines/mixedmodels.jl"
      fi ;;
    rust)
      echo ">> glmm (Rust) -> results/glmm_{empirical,simulated}/"
      if ! command -v cargo >/dev/null; then
        echo "   skipped: cargo not found" >&2
      else
        $PIN cargo run --quiet --release --manifest-path "$ROOT/../Cargo.toml" \
          -p validation --example validation_fit
      fi ;;
    py)
      echo ">> glmm python port -> results/glmm_python_*/"
      # The repo's own venv first (python/venv, where `maturin develop --release`
      # installs the wheel editable), else whatever python3 has glmm importable.
      # The wheel MUST be a --release build: a debug kernel would report the port's
      # "overhead" as a codegen artifact an order of magnitude too large.
      # dirname "$ROOT" (already absolute), not "$ROOT/..": a literal ".." in
      # sys.executable makes the venv's site.py warn about an unexpected sys.prefix.
      PY="$(dirname "$ROOT")/python/venv/bin/python"
      [[ -x "$PY" ]] || PY="$(command -v python3 || true)"
      if [[ -z "$PY" ]] || ! "$PY" -c 'import glmm' 2>/dev/null; then
        echo "   skipped: no python with the glmm wheel installed (see README setup)" >&2
      else
        $PIN "$PY" "$ROOT/engines/glmm_python.py"
      fi ;;
    glmm_r)
      echo ">> glmm R port -> results/glmm_r_*/"
      # Same kernel as the Rust/Python engines, reached through the fastglmm R
      # package (extendr wrapper). No venv step -- the package is installed in the
      # R library. Skip cleanly if it is not, so a machine without it still runs
      # the rest.
      if ! Rscript -e 'if (!requireNamespace("fastglmm", quietly=TRUE)) quit(status=1)' 2>/dev/null; then
        echo "   skipped: fastglmm R package not installed (see README setup)" >&2
      else
        $PIN Rscript "$ROOT/engines/glmm_r.R"
      fi ;;
    *)
      echo "unknown engine: $e" >&2; exit 2 ;;
  esac
done

echo ">> compare"
Rscript "$ROOT/compare.R"
