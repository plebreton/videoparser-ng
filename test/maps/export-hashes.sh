#!/usr/bin/env bash
# Print a hash of the QP, MV and bits exports of each test clip, to check that
# a refactoring does not change them. Usage: test/maps/export-hashes.sh <video-parser>
set -euo pipefail
parser="$1"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for clip in test/test-libx264.mp4 test/test-libx264-bitflip.ts test/test-libx265.mp4 \
            test/test-libvpx-vp9.mp4 test/test-libaom-av1.mp4; do
  "$parser" --export-qp "$tmp/qp" --export-mv "$tmp/mv" --export-bits "$tmp/bits" "$clip" > "$tmp/out" 2>/dev/null
  echo "$clip $(shasum "$tmp/qp" "$tmp/mv" "$tmp/bits" "$tmp/out" | awk '{print $1}' | tr '\n' ' ')"
done
