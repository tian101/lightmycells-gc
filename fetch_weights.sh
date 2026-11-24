#!/usr/bin/env bash
# Collect the four checkpoints into weights/ with the names the entrypoint expects.
#
# Run this on the machine that has the training checkpoints (the SSH alias is
# configurable), or point CHECK_DIR at a local directory that holds them.
set -euo pipefail

HOST="${HOST:-jupyter-gpu}"
CHECK_DIR="${CHECK_DIR:-/mnt/data0/lixt/pix/check}"
DEST="$(cd "$(dirname "$0")" && pwd)/weights"

mkdir -p "$DEST"

# channel : checkpoint directory
MAP=(
  "nucleus:lmc_nucleus_1111_e900_sfmfull_v4_cos1100"
  "mitochondria:lmc_mitochondria_1111_e800_sfmfull_v2"
  "actin:lmc_actin_1118_e800_sfmfull_v4"
  "tubulin:lmc_tubulin_1118_e800_sfmfull_v4"
)

if [ -d "$CHECK_DIR" ]; then
  for entry in "${MAP[@]}"; do
    ch="${entry%%:*}"; run="${entry#*:}"
    src="$CHECK_DIR/$run/latest_net_G.pth"
    [ -f "$src" ] || { echo "missing: $src" >&2; exit 1; }
    cp -f "$src" "$DEST/$ch.pth"
    echo "copied $ch.pth <- $run"
  done
else
  for entry in "${MAP[@]}"; do
    ch="${entry%%:*}"; run="${entry#*:}"
    scp -q "$HOST:$CHECK_DIR/$run/latest_net_G.pth" "$DEST/$ch.pth"
    echo "pulled $ch.pth <- $run"
  done
fi

echo
ls -lh "$DEST"
du -sh "$DEST"
