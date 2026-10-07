#!/usr/bin/env bash
# Check the exports of H.264 and HEVC streams with film grain: the decoder
# outputs a separate film grain frame, which must carry the same QP, MV and bits grids
# as the stream without film grain. Needs an ffmpeg program in PATH (stream copy
# only). Usage: test/maps/film-grain-check.sh <video-parser> <maps-check>
set -euo pipefail
parser="$1"
maps_check="$2"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
failures=0
for spec in h264:test/test-libx264.mp4 hevc:test/test-libx265.mp4; do
  codec="${spec%%:*}"
  clip="${spec#*:}"
  ffmpeg -v error -i "$clip" -c copy -bsf:v "${codec}_mp4toannexb" -f "$codec" "$tmp/plain.$codec"
  python3 "$(dirname "$0")/film-grain-sei.py" "$codec" "$tmp/plain.$codec" "$tmp/grain.$codec"
  for v in plain grain; do
    "$parser" --export-qp "$tmp/$v.qp" --export-mv "$tmp/$v.mv" --export-bits "$tmp/$v.bits" "$tmp/$v.$codec" > /dev/null 2>&1
  done
  if [ -s "$tmp/plain.qp" ] && [ -s "$tmp/plain.mv" ] && [ -s "$tmp/plain.bits" ] &&
     cmp -s "$tmp/plain.qp" "$tmp/grain.qp" && cmp -s "$tmp/plain.mv" "$tmp/grain.mv" &&
     cmp -s "$tmp/plain.bits" "$tmp/grain.bits"; then
    echo "$clip: film grain exports equal the exports without film grain: ok"
  else
    echo "$clip: film grain exports differ from the exports without film grain: FAIL"
    failures=$((failures + 1))
  fi
  "$maps_check" "$tmp/grain.$codec" 2>/dev/null | grep -v container_bit_rate || failures=$((failures + 1))
done
exit $((failures > 0))
