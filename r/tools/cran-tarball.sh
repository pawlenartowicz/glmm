#!/usr/bin/env sh
# Build the self-contained fastglmm source tarball (R-port spec §6).
#
# In the repo, src/rust depends on ../../../glmm-r, which depends on ../
# (the glmm kernel) — path deps that do not exist inside a built tarball. This
# script stages a copy of r/ with both crates materialized under
# src/rust/local/ and the deps repointed, so `R CMD build` / `R CMD check
# --as-cran` work on the result. The repo tree is never touched.
#
#   tools/cran-tarball.sh [--vendor]
#
# --vendor additionally packs the crates.io dependency tree
# (vendor.tar.xz + vendor-config.toml, consumed by src/Makevars for offline
# CRAN builds — measured 2026-07-15: 137 crates, ~13 MB; the 6 MB no-gemm
# faer config was rejected, it drops the BLAS-3 backend). Without it the
# tarball builds online (cargo fetches), which is all a local check needs.
#
# The vendored tree lives only in the produced tarball, never in git
# (spec §6). Plan gate 4 (extendr vendoring vs current CRAN policy) is
# verified by running --as-cran on the --vendor artifact.
set -eu

want_vendor=0
[ "${1:-}" = "--vendor" ] && want_vendor=1

repo=$(cd "$(dirname "$0")/../.." && pwd)   # GLMM repo root
out=$(pwd)
# glmm's version — pins the `cargo package` output name and the glmm-r dep
# below. Read from the manifest so a version bump needs no edit here.
ver=$(sed -n 's/^version = "\(.*\)"/\1/p' "$repo/Cargo.toml" | head -1)
stage="$repo/target/cran-stage"
pkg="$stage/fastglmm"

rm -rf "$stage"
mkdir -p "$stage"
cp -r "$repo/r" "$pkg"
rm -rf "$pkg/src/rust/target" "$pkg/src/rust/vendor" \
       "$pkg/src/.cargo" "$pkg/src/Makevars" "$pkg/src/Makevars.win"
local_dir="$pkg/src/rust/local"
mkdir -p "$local_dir"

# glmm: `cargo package` resolves the workspace-inherited manifest fields and
# prunes to the published file set — exactly the crates.io artifact.
(cd "$repo" && cargo package -p glmm --no-verify --allow-dirty >/dev/null)
tar -xzf "$repo/target/package/glmm-$ver.crate" -C "$local_dir"
mv "$local_dir"/glmm-* "$local_dir/glmm"
# The crate currently packages the whole repo tree; drop what the staticlib
# build cannot need and what R CMD check flags (hidden files, the R port
# itself). parity/ stays: the manifest's [[example]] targets point into it.
rm -rf "$local_dir/glmm/r" "$local_dir/glmm/.github" \
       "$local_dir/glmm/.cargo_vcs_info.json"

# glmm-r: unpackageable while this glmm version is unpublished (cargo package
# verifies version deps against the registry), so copy + de-workspace its
# manifest by hand. Anchored edits — mirrors glmm-r/Cargo.toml, change
# together.
cp -r "$repo/glmm-r" "$local_dir/glmm-r"
rm -rf "$local_dir/glmm-r/target"
sed -i \
  -e 's/^edition\.workspace = true$/edition = "2021"/' \
  -e 's/^rust-version\.workspace = true$/rust-version = "1.85"/' \
  -e 's/^license\.workspace = true$/license = "GPL-3.0-or-later"/' \
  -e "s/^glmm = .*\$/glmm = { version = \"$ver\" }/" \
  -e '/^\[lints\]$/,/^workspace = true$/d' \
  "$local_dir/glmm-r/Cargo.toml"

# Repoint the staticlib crate at the local copies; the [patch] makes the
# version dep resolve to local/glmm even though that version may not be
# on crates.io.
sed -i 's|glmm-r      = { path = "../../../glmm-r" }|glmm-r      = { path = "local/glmm-r" }|' \
  "$pkg/src/rust/Cargo.toml"
cat >> "$pkg/src/rust/Cargo.toml" <<'EOF'

[patch.crates-io]
glmm = { path = "local/glmm" }
EOF
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
