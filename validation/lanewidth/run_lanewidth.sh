#!/usr/bin/env bash
# Lane-width sensitivity harness.
#
# Measures how much of a fit result is SIMD-lane-width / summation-order
# dependent by running the crate's tests twice from a scratch copy of the
# source tree: once against the normal, registry pulp =0.22.2 (whatever
# lane width this host's runtime CPU detection picks), and once against a
# vendored, patched copy of the SAME pulp version with `Arch::new()` forced
# to always return `Scalar` (lane width 1), so every pulp::WithSimd kernel
# takes the scalar path instead. See README.md next to this script for the
# full design rationale and how to interpret the diff between the two runs.
#
# HARD CONSTRAINT (RULE 4): the patch is applied ONLY inside a
# scratch copy this script creates. It never touches the real tree's
# Cargo.toml/Cargo.lock, and the real pulp pin (`=0.22.2` in
# GLMM/Cargo.toml) is never edited. Running this script must leave the real
# tree bit-for-bit as it found it.
#
#   ./run_lanewidth.sh                       default filter (NB/sparse fit tests)
#   ./run_lanewidth.sh test_name_a test_b     restrict the cargo test filter(s)
#   SCRATCH=/some/dir ./run_lanewidth.sh      pin the scratch dir (default: mktemp -d)
#   KEEP_SCRATCH=1 ./run_lanewidth.sh         don't delete the scratch dir on exit
#
# Output: a timestamped results/<ts>/ directory next to this script holding
# the two lane probes, the two cargo-test logs, and a summary.txt. Nothing
# under results/ is committed (see .gitignore); it is regenerated per run,
# same convention as validation/results/.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_ROOT="$(cd "$SELF/../.." && pwd)"          # GLMM/
PATCH_FILE="$SELF/pulp-0.22.2-scalar-force.patch"
PULP_VERSION="0.22.2"

[[ -f "$PATCH_FILE" ]] || { echo "missing patch file: $PATCH_FILE" >&2; exit 2; }
[[ -f "$CRATE_ROOT/Cargo.toml" ]] || { echo "cannot find crate root from $SELF" >&2; exit 2; }

# Default test filter: the sparse NB fit tests this harness was originally
# built to check (the crate's most lane-width-sensitive pin, per its own
# doc comment in src/sparse/tests.rs). Override with positional args.
DEFAULT_FILTERS=(fit_sparse_nb_glmm_is_pinned sparse_weighted_nb_matches_replicated)
if [[ $# -gt 0 ]]; then
  FILTERS=("$@")
else
  FILTERS=("${DEFAULT_FILTERS[@]}")
fi

# A user-supplied SCRATCH that already exists gets rm -rf'd on exit by the
# cleanup trap below — refuse rather than silently delete someone's
# pre-existing directory. The mktemp default path is always fresh, so this
# only guards the override.
if [[ -n "${SCRATCH:-}" && -e "$SCRATCH" ]]; then
  echo "refusing to run: SCRATCH=$SCRATCH already exists, and this script" >&2
  echo "removes it (rm -rf) on exit — point SCRATCH at a path that does not" >&2
  echo "exist yet, or omit SCRATCH to use a fresh mktemp directory" >&2
  exit 2
fi

SCRATCH="${SCRATCH:-$(mktemp -d -t glmm-lanewidth)}"
KEEP_SCRATCH="${KEEP_SCRATCH:-0}"
mkdir -p "$SCRATCH"
echo "scratch dir: $SCRATCH"

cleanup() {
  if [[ "$KEEP_SCRATCH" != "1" ]]; then
    rm -rf "$SCRATCH"
  else
    echo "KEEP_SCRATCH=1 set, leaving scratch dir at: $SCRATCH"
  fi
}
trap cleanup EXIT

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$SELF/results/$TS"
mkdir -p "$RESULTS_DIR"

HOST_ARCH="$(uname -m)"
HOST_OS="$(uname -s)"
echo "host: $HOST_OS/$HOST_ARCH"

# --- 1. copy the crate source into two scratch trees (normal, scalar) -----
# Excludes: build artifacts and venvs only, never source. The two copies
# start identical; only scalar/Cargo.toml gets the extra [patch.crates-io]
# table appended in step 4.
echo "== copying crate source into scratch =="
RSYNC_EXCLUDES=(--exclude=/target --exclude=/python/venv --exclude=/python/build
                --exclude=/python/*.egg-info --exclude=/python/.pytest_cache
                --exclude=.DS_Store --exclude=/validation/results
                --exclude=/validation/lanewidth/results)
rsync -a "${RSYNC_EXCLUDES[@]}" "$CRATE_ROOT/" "$SCRATCH/normal/"
cp -R "$SCRATCH/normal" "$SCRATCH/scalar"

# --- 2. get pulp 0.22.2 source: prefer the local registry cache, else vendor ---
echo "== fetching pulp $PULP_VERSION source =="
VENDOR_DIR="$SCRATCH/vendor/pulp-$PULP_VERSION"
mkdir -p "$SCRATCH/vendor"
CACHE_SRC="$(ls -d "$HOME"/.cargo/registry/src/*/pulp-"$PULP_VERSION" 2>/dev/null | head -1 || true)"
if [[ -n "$CACHE_SRC" ]]; then
  cp -R "$CACHE_SRC" "$VENDOR_DIR"
  SOURCE_METHOD="local registry cache ($CACHE_SRC)"
else
  # No local cache: fetch it the honest way, via cargo vendor from the
  # normal scratch copy (network access expected/fine per the task brief).
  ( cd "$SCRATCH/normal" && cargo vendor "$SCRATCH/vendor-fetch" >/dev/null )
  [[ -d "$SCRATCH/vendor-fetch/pulp-$PULP_VERSION" ]] \
    || { echo "cargo vendor did not produce pulp-$PULP_VERSION" >&2; exit 3; }
  cp -R "$SCRATCH/vendor-fetch/pulp-$PULP_VERSION" "$VENDOR_DIR"
  SOURCE_METHOD="cargo vendor (network fetch)"
fi
echo "pulp source via: $SOURCE_METHOD"

# --- 3. apply the scalar-forcing patch to the vendored copy ONLY ----------
echo "== applying scalar-forcing patch =="
patch -p1 -d "$VENDOR_DIR" < "$PATCH_FILE"
grep -q "LANEWIDTH-HARNESS" "$VENDOR_DIR/src/aarch64.rs" "$VENDOR_DIR/src/x86.rs" "$VENDOR_DIR/src/wasm.rs" \
  || { echo "patch marker missing after apply — patch did not land as expected" >&2; exit 3; }

# --- 4. wire the patched pulp into the SCALAR scratch copy only -----------
# [patch.crates-io] with a path override — not cargo vendor's full
# source-replacement config — so only pulp is redirected; every other
# dependency still resolves normally. This table is appended to the scratch
# copy's Cargo.toml and is NEVER written to the real tree.
cat >> "$SCRATCH/scalar/Cargo.toml" <<EOF

[patch.crates-io]
pulp = { path = "$VENDOR_DIR" }
EOF

# --- 5. lane-width probe: what does Arch::new() actually resolve to, and ---
#        does the pinned sparse-NB fit's beta/theta actually move? ----------
# `lanewidth_probe.rs` (committed next to this script) is staged as an
# example into BOTH scratch copies (never into the real tree). It prints
# pulp::Arch::new()'s Debug repr (the printed-dispatch-marker acceptance
# evidence) and refits the same sparse NB design as
# `fit_sparse_nb_glmm_is_pinned` (src/sparse/tests.rs), so a `diff` of the
# two probe outputs shows the actual numeric movement this harness exists
# to measure — see README.md for a worked example.
for tree in normal scalar; do
  mkdir -p "$SCRATCH/$tree/examples"
  cp "$SELF/lanewidth_probe.rs" "$SCRATCH/$tree/examples/lanewidth_probe.rs"
done

# --release: the crate's own code (not just its dependencies) is only
# optimized under [profile.release] or [profile.test] — plain `cargo run`
# uses [profile.dev], where the sparse NB GLMM fit is documented (Cargo.toml,
# profile.test comment) as 30-60x slower, "tens of minutes" instead of ~24s.
echo "== running lane-width probe (normal) =="
( cd "$SCRATCH/normal" && cargo run --release --quiet --example lanewidth_probe ) \
  | tee "$RESULTS_DIR/probe_normal.txt"

echo "== running lane-width probe (scalar-forced) =="
( cd "$SCRATCH/scalar" && cargo run --release --quiet --example lanewidth_probe ) \
  | tee "$RESULTS_DIR/probe_scalar.txt"

DISPATCH_NORMAL="$(grep '^pulp::Arch::new()' "$RESULTS_DIR/probe_normal.txt" || true)"
DISPATCH_SCALAR="$(grep '^pulp::Arch::new()' "$RESULTS_DIR/probe_scalar.txt" || true)"

# --- 6. run the caller-specified test filter in both trees ----------------
echo "== running test filter in normal tree: ${FILTERS[*]} =="
NORMAL_STATUS=0
( cd "$SCRATCH/normal" && cargo test --lib -- "${FILTERS[@]}" --nocapture ) \
  > "$RESULTS_DIR/test_normal.log" 2>&1 || NORMAL_STATUS=$?
tail -n 20 "$RESULTS_DIR/test_normal.log"

echo "== running test filter in scalar-forced tree: ${FILTERS[*]} =="
SCALAR_STATUS=0
( cd "$SCRATCH/scalar" && cargo test --lib -- "${FILTERS[@]}" --nocapture ) \
  > "$RESULTS_DIR/test_scalar.log" 2>&1 || SCALAR_STATUS=$?
tail -n 20 "$RESULTS_DIR/test_scalar.log"

# --- 7. summary -------------------------------------------------------------
{
  echo "lane-width harness run — $TS"
  echo "host: $HOST_OS/$HOST_ARCH"
  echo "pulp source: $SOURCE_METHOD"
  echo "test filter(s): ${FILTERS[*]}"
  echo
  echo "-- lane-width dispatch marker (pulp::Arch::new() Debug repr) --"
  echo "normal:        $DISPATCH_NORMAL"
  echo "scalar-forced: $DISPATCH_SCALAR"
  if [[ "$DISPATCH_NORMAL" == "$DISPATCH_SCALAR" ]]; then
    echo "WARNING: dispatch marker identical between runs — scalar forcing may not"
    echo "         have taken effect (e.g. host was already scalar-only)."
  fi
  echo
  echo "-- full probe output (Arch + refit beta/se/varcorr/dispersion) --"
  echo "normal:        $RESULTS_DIR/probe_normal.txt"
  echo "scalar-forced: $RESULTS_DIR/probe_scalar.txt"
  echo "(diff the two files for the actual numeric movement)"
  echo
  echo "-- cargo test exit status --"
  echo "normal:        exit $NORMAL_STATUS"
  echo "scalar-forced: exit $SCALAR_STATUS"
  echo "(a non-zero scalar-forced status on a tightly-pinned test is EXPECTED"
  echo " evidence of lane-width drift, not a harness bug — see README.md)"
  echo
  echo "full logs: $RESULTS_DIR/{test_normal,test_scalar}.log"
} | tee "$RESULTS_DIR/summary.txt"

echo
echo "results written to: $RESULTS_DIR"
