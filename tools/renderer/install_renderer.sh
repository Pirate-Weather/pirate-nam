#!/usr/bin/env bash
# Install the pinned map renderer for this tree: tools/renderer/bin/rw_wrfbatch + assets/basemap beside it
# (namgpu/io/maps.py PINNED_RENDERER; render law: weather maps come from rw_wrfbatch).
#
#   tools/renderer/install_renderer.sh --from-src <gpuwm checkout>   # build (offline cargo)
#   tools/renderer/install_renderer.sh --from-bin <dir with rw_wrfbatch and assets/basemap>   # copy a built one
#
# --from-src copies gpuwm's tools/ (no target/) to a scratch dir, applies every tools/renderer/*.patch, builds
# rw-wrfbatch with all cores (cargo build --release --locked --offline -p rw-wrfbatch) and installs it. The gpuwm
# cut must be 42631c9f83e or later (dateline frames). Patches:
#   01-cf_grid-recenter-order.patch  CfGrid::recenter undoes open()'s -180/180 seam rotation before re-seaming, so a
#                                 curvilinear row across 180 keeps its column order (without it the cells east of
#                                 180 were drawn from the wrong neighbours: the stepped strip on the Gabrielle
#                                 37S 175E sheets, 2026-11-05).
#   02-cf_scale-mask-below.patch  a CF product row's scale may carry mask_below: values under it are blank instead
#                                 of clamped into the first colour (a dry 1 h / 6 h window painted a domain-wide
#                                 first band that read as rain).
#   03-cf_row-note-subtitle.patch a CF product row may carry "note", appended to the panel subtitle after the valid
#                                 time (the data credit CC BY sources need on every map; in the title it widened
#                                 the panel, cut the valid time and broke sheet composition).
#   04-ensemble-sheet-pad-subtitle.patch  --ensemble-sheet pads panels of two sizes to the largest (top-left, the panel's
#                                 corner colour; no resampling) instead of refusing the sheet, and takes an optional
#                                 request "subtitle" drawn as given (the sheet header was "valid <valid_time> UTC", so a
#                                 header with init and a data credit read "... CC BY 4.0 UTC"; a 4 px wider panel killed
#                                 the Gabrielle nest's whole sheets stage, 2026-11-05).
# Basemap: Natural Earth 10m/110m (public domain) and US Census cb_2023 counties (public domain), from the cut's
# tools/rustwx/assets/basemap with its THIRD-PARTY-LICENSES.txt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/bin"
mode="${1:-}"; src="${2:-}"
[ -n "$mode" ] && [ -n "$src" ] || { sed -n 2,8p "$0"; exit 2; }
mkdir -p "$DEST"
case "$mode" in
  --from-bin)
    [ -x "$src/rw_wrfbatch" ] || { echo "no rw_wrfbatch in $src" >&2; exit 2; }
    [ -d "$src/assets/basemap" ] || { echo "no assets/basemap in $src" >&2; exit 2; }
    cp -f "$src/rw_wrfbatch" "$DEST/rw_wrfbatch"
    rm -rf "$DEST/assets"; mkdir -p "$DEST/assets"; cp -a "$src/assets/basemap" "$DEST/assets/basemap"
    [ -f "$src/REVISION.txt" ] && cp -f "$src/REVISION.txt" "$DEST/REVISION.txt"
    echo "copied from $src $(date -u +%FT%TZ)" >> "$DEST/REVISION.txt" ;;
  --from-src)
    # the rustwx workspace has path dependencies on its sibling crates under gpuwm's tools/ (grib1_bridge,
    # netcdf-writer, radar_core, ...), so the whole tools/ folder is copied (no target/ dirs, ~1.4 GB)
    tools="$src/tools"; [ -d "$tools/rustwx" ] || { echo "$src is not a gpuwm checkout (no tools/rustwx)" >&2; exit 2; }
    rw="$tools/rustwx"
    work="${NAMGPU_RENDERER_BUILD:-${TMPDIR:-/tmp}/namgpu-renderer-build}"
    mkdir -p "$work"
    rsync -a --delete --exclude "/*/target/" --exclude __pycache__ "$tools/" "$work/tools/"
    # files outside tools/ that the crates read at build time: the engine version (rw-wrfbatch build.rs) and
    # gpuwm/authorities, gpuwm/data, gpuwm/physics_consumer_export_v1.json (include_str! in static-fields etc.)
    cp -f "$src/pyproject.toml" "$work/pyproject.toml"
    mkdir -p "$work/gpuwm"
    rsync -a --delete "$src/gpuwm/authorities/" "$work/gpuwm/authorities/"
    rsync -a --delete --exclude __pycache__ "$src/gpuwm/data/" "$work/gpuwm/data/"
    cp -f "$src/gpuwm/physics_consumer_export_v1.json" "$work/gpuwm/"
    for p in "$HERE"/*.patch; do (cd "$work/tools/rustwx" && patch -p1 --forward < "$p"); done
    (cd "$work/tools/rustwx" && cargo build --release --locked --offline -p rw-wrfbatch -j "$(nproc)")
    cp -f "$work/tools/rustwx/target/release/rw_wrfbatch" "$DEST/rw_wrfbatch"
    rm -rf "$DEST/assets"; mkdir -p "$DEST/assets"; cp -a "$rw/assets/basemap" "$DEST/assets/basemap"
    rev="$(git -C "$rw" rev-parse --short=11 HEAD 2>/dev/null || echo unknown)"
    echo "rw_wrfbatch built $(date -u +%FT%TZ) from $rw ($rev) + $(cd "$HERE" && ls *.patch | tr '\n' ' ')" > "$DEST/REVISION.txt" ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
"$DEST/rw_wrfbatch" --help >/dev/null 2>&1 || true
echo "installed $DEST/rw_wrfbatch"; cat "$DEST/REVISION.txt"
