#!/usr/bin/env sh
# Build the self-contained fastglmm source tarball.
#
# In the repo, src/rust depends on ../../rust (the glmm-r crate), which depends
# on ../../glmm-bind, which depends on ../glmm-rs (the glmm kernel): path deps
# that do not exist inside a built tarball. This script stages a copy of glmm-r/
# with the three crates materialized under src/rust/local/ and the deps
# repointed, so `R CMD build` / `R CMD check --as-cran` work on the result. The
# repo tree is never touched.
#
#   tools/cran-tarball.sh [--vendor]
#
# --vendor additionally packs the crates.io dependency tree
# (vendor.tar.xz + vendor-config.toml, consumed by src/Makevars for offline
# CRAN builds — measured 2026-07-15: 137 crates, ~13 MB; the 6 MB no-gemm
# faer config is not used because it drops the BLAS-3 backend). Without it the
# tarball builds online (cargo fetches), which is all a local check needs.
#
# The vendored tree lives only in the produced tarball, never in git.
# Extendr vendoring against current CRAN policy is
# verified by running --as-cran on the --vendor artifact.
set -eu

want_vendor=0
[ "${1:-}" = "--vendor" ] && want_vendor=1

repo=$(cd "$(dirname "$0")/../.." && pwd)   # GLMM repo root
out=$(pwd)
# glmm's version: pins the `cargo package` output name below. Read from the
# manifest so a version bump needs no edit here.
ver=$(sed -n 's/^version = "\(.*\)"/\1/p' "$repo/glmm-rs/Cargo.toml" | head -1)
stage="$repo/target/cran-stage"
pkg="$stage/fastglmm"

rm -rf "$stage"
mkdir -p "$stage"
cp -r "$repo/glmm-r" "$pkg"
# The references tests/testthat/test-robust.R reads from the repository's
# validation/ tree, which the tarball does not ship. Mirrors the names used
# in test-robust.R - change together.
fixtures="$pkg/tests/testthat/fixtures"
mkdir -p "$fixtures"
for f in cr2_ri_s10 cr2_rs_s10 hc2_n20 hc2_n100; do
  cp "$repo/validation/data/inference/$f.csv" "$fixtures/"
done
for f in cr2_lme_ri_estimated_s10 hc2_n20; do
  cp "$repo/validation/goldens/inference/cr2/$f.json" "$fixtures/"
done
rm -rf "$pkg/rust" "$pkg/src/rust/target" "$pkg/src/rust/vendor" \
       "$pkg/src/.cargo" "$pkg/src/Makevars" "$pkg/src/Makevars.win"
local_dir="$pkg/src/rust/local"
mkdir -p "$local_dir"

# glmm: `cargo package` resolves the workspace-inherited manifest fields and
# prunes to the published file set — exactly the crates.io artifact.
(cd "$repo" && cargo package -p glmm --no-verify --allow-dirty >/dev/null)
tar -xzf "$repo/target/package/glmm-$ver.crate" -C "$local_dir"
mv "$local_dir"/glmm-* "$local_dir/glmm"
# Drop the hidden file R CMD check flags.
rm -rf "$local_dir/glmm/.cargo_vcs_info.json"

# glmm-bind and glmm-r: unpublished (publish = false), so copy + de-workspace
# their manifests by hand. Anchored edits: mirrors glmm-bind/Cargo.toml and
# glmm-r/rust/Cargo.toml, change together. Each dep line has only its path
# rewritten, so glmm-bind's `features = ["orchestrate"]` rides along (without
# it the staticlib build fails with `unresolved import glmm::orchestrate`).
cp -r "$repo/glmm-bind" "$local_dir/glmm-bind"
cp -r "$repo/glmm-r/rust" "$local_dir/glmm-r"
for crate in glmm-bind glmm-r; do
  rm -rf "${local_dir:?}/$crate/target"
  sed -i \
    -e 's/^edition\.workspace = true$/edition = "2021"/' \
    -e 's/^rust-version\.workspace = true$/rust-version = "1.85"/' \
    -e 's/^license\.workspace = true$/license = "LGPL-3.0-or-later"/' \
    -e 's|^glmm = { path = "\.\./glmm-rs", |glmm = { path = "../glmm", |' \
    -e 's|^glmm-bind = { path = "\.\./\.\./glmm-bind" }$|glmm-bind = { path = "../glmm-bind" }|' \
    -e '/^\[lints\]$/,/^workspace = true$/d' \
    "$local_dir/$crate/Cargo.toml"
done

# Repoint the staticlib crate at the local copies.
sed -i 's|glmm-r      = { path = "../../rust" }|glmm-r      = { path = "local/glmm-r" }|' \
  "$pkg/src/rust/Cargo.toml"
# Refresh the lockfile for the rewritten manifest.
(cd "$pkg/src/rust" && cargo update --workspace --quiet)

if [ "$want_vendor" = 1 ]; then
  (
    cd "$pkg/src/rust"
    cargo vendor vendor > vendor-config.toml
    # cargo prints an absolute directory path; Makevars unpacks beside the
    # manifest, so relativize it.
    sed -i 's|^directory = .*|directory = "vendor"|' vendor-config.toml
    tar -cJf vendor.tar.xz vendor
    rm -rf vendor
  )
fi

(cd "$out" && R CMD build "$pkg")
echo "staged tree left at $pkg"
