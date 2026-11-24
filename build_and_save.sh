#!/usr/bin/env bash
# Build the container and export it for upload to the challenge platform.
#
#   ./build_and_save.sh            # build, smoke test, save lightmycells-v4.tar.gz
#   TAG=1.1 ./build_and_save.sh    # explicit tag
#   SKIP_TEST=1 ./build_and_save.sh
set -euo pipefail

TAG="${TAG:-1}"
IMAGE="${IMAGE:-lightmycells-v4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

for ch in nucleus mitochondria actin tubulin; do
  [ -f "weights/$ch.pth" ] || { echo "weights/$ch.pth is missing - run ./fetch_weights.sh" >&2; exit 1; }
done

echo "== build $IMAGE:$TAG =="
docker build --platform linux/amd64 -t "$IMAGE:$TAG" .

if [ -z "${SKIP_TEST:-}" ]; then
  echo "== smoke test =="
  tmp="$(mktemp -d)"
  mkdir -p "$tmp/in/images/organelles-transmitted-light-ome-tiff" "$tmp/out"
  # 64x64 synthetic plane: proves the plumbing (weights load, 4 files written)
  python - "$tmp/in/images/organelles-transmitted-light-ome-tiff/smoke.tif" <<'PY'
import sys
import numpy as np, tifffile
y, x = np.mgrid[0:512, 0:512]
plane = (32768 + 12000 * np.sin(x / 23.0) * np.cos(y / 31.0)).astype(np.uint16)
tifffile.imwrite(sys.argv[1], plane, ome=True, photometric="minisblack", metadata={"axes": "YX"})
PY
  docker run --rm --gpus all -v "$tmp/in:/input:ro" -v "$tmp/out:/output" "$IMAGE:$TAG"
  found=$(find "$tmp/out/images" -name '*.tif' | wc -l)
  echo "smoke test wrote $found output file(s)"
  [ "$found" -eq 4 ] || { echo "expected 4 outputs, got $found" >&2; exit 1; }
  rm -rf "$tmp"
fi

echo "== save =="
out="$HERE/$IMAGE.tar.gz"
docker save "$IMAGE:$TAG" | gzip -1 > "$out"
ls -lh "$out"
echo
echo "Upload $out via Container Images -> Upload a Container."
